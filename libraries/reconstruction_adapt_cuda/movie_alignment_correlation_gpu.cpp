/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "reconstruction_adapt_cuda/movie_alignment_correlation_gpu.h"

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::defineParams() {
    AProgMovieAlignmentCorrelation<T>::defineParams();
    this->addParamsLine("  [--device <dev=0>]                 : GPU device to use. 0th by default");
    this->addParamsLine("  [--storage <fn=\"\">]              : Path to file that can be used to store results of the benchmark");
    this->addParamsLine("  [--controlPoints <x=6> <y=6> <t=5>]: Number of control points (including end points) used for defining the BSpline");
    this->addParamsLine("  [--patches <x=10> <y=10>]          : Number of patches to use for local alignment estimation");
    this->addParamsLine("  [--patchesAvg <avg=3>]             : Number of near frames used for averaging a single patch");
    this->addParamsLine("  [--oBSpline <fn=\"\">]             : Path to file that can be used to store BSpline coefficients");

    this->addExampleLine(
                "xmipp_cuda_movie_alignment_correlation -i movie.xmd --oaligned alignedMovie.stk --oavg alignedMicrograph.mrc --device 0");
    this->addSeeAlsoLine("xmipp_movie_alignment_correlation");
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::show() {
    AProgMovieAlignmentCorrelation<T>::show();
    std::cout << "gpu set: " << gpu.has_value() << std::endl;
    std::cout << "Device:              " << gpu.value().device() << " (" << gpu.value().UUID() << ")" << std::endl;
    std::cout << "Benchmark storage    " << (storage.empty() ? "Default" : storage) << std::endl;
    std::cout << "Control points:      " << localAlignmentControlPoints << std::endl;
    std::cout << "Patches:             " << localAlignPatches.first << " x " << localAlignPatches.second << std::endl;
    std::cout << "Patches avg:         " << patchesAvg << std::endl;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::readParams() {
    AProgMovieAlignmentCorrelation<T>::readParams();

    // read GPU
    int device = this->getIntParam("--device");
    assert(device >= 0);
    gpu = std::move(core::optional<GPU>(GPU(device)));

    // read permanent storage
    storage = this->getParam("--storage");

    // read control points
    Dimensions cPoints(
            this->getIntParam("--controlPoints", 0),
            this->getIntParam("--controlPoints", 1),
            1,
            this->getIntParam("--controlPoints", 2));
    assert(cPoints.x() >= 3 && cPoints.y() >= 3 && cPoints.n() >= 3);
    localAlignmentControlPoints = cPoints;

    // read patches
    localAlignPatches = std::make_pair(
            this->getIntParam("--patches", 0),
            this->getIntParam("--patches", 1));
    assert(localAlignPatches.first > 0 && localAlignPatches.second > 0);

    // read patch averaging
    patchesAvg = this->getIntParam("--patchesAvg");
    assert(patchesAvg >= 1);

    // read BSpline coefficients storage
    fnBSplinePath = this->getParam("--oBSpline");
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getSettingsOrBenchmark(
        const Dimensions &d, size_t extraMem, bool crop) {
    auto optSetting = getStoredSizes(d, crop);
    FFTSettings<T> result =
            optSetting ?
                    optSetting.value() : runBenchmark(d, extraMem, crop);
    if (!optSetting) {
        storeSizes(d, result, crop);
    }
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getMovieSettings(
        const MetaData &movie, bool optimize) {
    Image<T> frame;
    int noOfImgs = this->nlast - this->nfirst + 1;
    this->loadFrame(movie, movie.firstObject(), frame);
    Dimensions dim(frame.data.xdim, frame.data.ydim, 1, noOfImgs);

    if (optimize) {
        int maxFilterSize = getMaxFilterSize(frame);
        return getSettingsOrBenchmark(dim, maxFilterSize, true);
    } else {
        return FFTSettings<T>(dim, 1, false);
    }
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCorrelationHint(
        const FFTSettings<T> &s,
        const std::pair<T, T> &downscale) {
    // we need odd size of the input, to be able to
    // compute FFT more efficiently (and e.g. perform shift by multiplication)
    auto scaleEven = [] (size_t v, T downscale) {
        return (int(v * downscale) / 2) * 2;
    };
    Dimensions result(scaleEven(s.dim.x(), downscale.first),
            scaleEven(s.dim.y(), downscale.second), s.dim.z(),
            (s.dim.n() * (s.dim.n() - 1)) / 2); // number of correlations);
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCorrelationSettings(
        const FFTSettings<T> &orig,
        const std::pair<T, T> &downscale) {
    auto hint = getCorrelationHint(orig, downscale);
    size_t correlationBufferSizeMB = gpu.value().lastFreeMem() / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

    return getSettingsOrBenchmark(hint, 2 * correlationBufferSizeMB, false);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getPatchSettings(
        const FFTSettings<T> &orig) {
    Dimensions hint(512, 512, // this should be a trade-off between speed and present signal
            // but check the speed to make sure
            orig.dim.z(), orig.dim.n());
    size_t correlationBufferSizeMB = gpu.value().lastFreeMem() / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

    return getSettingsOrBenchmark(hint, 2 * correlationBufferSizeMB, false);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getPatchesLocation(
        const std::pair<T, T> &borders,
        const Dimensions &movie, const Dimensions &patch) {
    size_t patchesX = localAlignPatches.first;
    size_t patchesY = localAlignPatches.second;
    T windowXSize = movie.x() - 2 * borders.first;
    T windowYSize = movie.y() - 2 * borders.second;
    T corrX = std::ceil(
            ((patchesX * patch.x()) - windowXSize) / (T) (patchesX - 1));
    T corrY = std::ceil(
            ((patchesY * patch.y()) - windowYSize) / (T) (patchesY - 1));
    T stepX = (T)patch.x() - corrX;
    T stepY = (T)patch.y() - corrY;
    std::vector<FramePatchMeta<T>> result;
    for (size_t y = 0; y < patchesY; ++y) {
        for (size_t x = 0; x < patchesX; ++x) {
            T tlx = borders.first + x * stepX; // Top Left
            T tly = borders.second + y * stepY;
            T brx = tlx + patch.x() - 1; // Bottom Right
            T bry = tly + patch.y() - 1; // -1 for indexing
            Point2D<T> tl(tlx, tly);
            Point2D<T> br(brx, bry);
            Rectangle<Point2D<T>> r(tl, br);
            result.emplace_back(
                    FramePatchMeta<T> { .rec = r, .id_x = x, .id_y =
                    y });
        }
    }
    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchData(const T *allFrames,
        const Rectangle<Point2D<T>> &patch, const AlignmentResult<T> &globAlignment,
        const Dimensions &movie, T *result) {
    size_t n = movie.n();
    auto patchSize = patch.getSize();
    auto copyPatchData = [&](size_t srcFrameIdx, size_t t, bool add) {
        size_t frameOffset = srcFrameIdx * movie.x() * movie.y();
        size_t patchOffset = t * patchSize.x * patchSize.y;
        // keep the shift consistent while adding local shift
        int xShift = std::round(globAlignment.shifts.at(srcFrameIdx).x);
        int yShift = std::round(globAlignment.shifts.at(srcFrameIdx).y);
        for (size_t y = 0; y < patchSize.y; ++y) {
            size_t srcY = patch.tl.y + y;
            if (yShift < 0) {
                srcY -= (size_t)std::abs(yShift); // assuming shift is smaller than offset
            } else {
                srcY += yShift;
            }
            size_t srcIndex = frameOffset + (srcY * movie.x()) + (size_t)patch.tl.x;
            if (xShift < 0) {
                srcIndex -= (size_t)std::abs(xShift);
            } else {
                srcIndex += xShift;
            }
            size_t destIndex = patchOffset + y * patchSize.x;
            if (add) {
                for (size_t x = 0; x < patchSize.x; ++x) {
                    result[destIndex + x] += allFrames[srcIndex + x];
                }
            } else {
                memcpy(result + destIndex, allFrames + srcIndex, patchSize.x * sizeof(T));
            }
        }
    };
    for (int t = 0; t < n; ++t) {
        // copy the data from specific frame
        copyPatchData(t, t, false);
        // add data from frames with lower indices
        // while averaging odd num of frames, use copy equally from previous and following frames
        // otherwise prefer following frames
        for (int b = 1; b <= ((patchesAvg - 1) / 2); ++b) {
            if (t >= b) {
                copyPatchData(t - b, t, true);
            }
        }
        // add data from frames with higher indices
        for (int f = 1; f <= (patchesAvg / 2); ++f) {
            if ((t + f) < n) {
                copyPatchData(t + f, t, true);
            }
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(const Dimensions &dim,
        const FFTSettings<T> &s, bool applyCrop) {
    UserSettings::get(storage).insert(*this,
            getKey(optSizeXStr, dim, applyCrop), s.dim.x());
    UserSettings::get(storage).insert(*this,
            getKey(optSizeYStr, dim, applyCrop), s.dim.y());
    UserSettings::get(storage).insert(*this,
            getKey(optBatchSizeStr, dim, applyCrop), s.batch);
    UserSettings::get(storage).insert(*this,
            getKey(minMemoryStr, dim, applyCrop), gpu.value().lastFreeMem());
    UserSettings::get(storage).store(); // write changes immediately
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(
        const Dimensions &dim, bool applyCrop) {
    size_t x, y, batch, neededMem;
    bool res = true;
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(optSizeXStr, dim, applyCrop), x);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(optSizeYStr, dim, applyCrop), y);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(optBatchSizeStr, dim, applyCrop), batch);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(minMemoryStr, dim, applyCrop), neededMem);
    res = res && neededMem <= gpu.value().checkFreeMem();
    if (res) {
        return core::optional<FFTSettings<T>>(
                FFTSettings<T>(x, y, 1, dim.n(), batch, true));
    } else {
        return core::optional<FFTSettings<T>>();
    }
}


template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::runBenchmark(const Dimensions &d,
        size_t extraMem, bool crop) {
    if (this->verbose) std::cerr << "Benchmarking cuFFT ..." << std::endl;
    // take additional memory requirement into account
    int x, y, batch;
    getBestFFTSize(d.n(), d.x(), d.y(), batch, crop, x, y, extraMem, this->verbose,
            gpu.value().device(), d.x() == d.y(), 10); // allow max 10% change

    return FFTSettings<T>(x, y, 1, d.n(), batch, true);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getMovieBorders(
        const AlignmentResult<T> &globAlignment, bool verbose) {
    T minX = std::numeric_limits<T>::max();
    T maxX = std::numeric_limits<T>::min();
    T minY = std::numeric_limits<T>::max();
    T maxY = std::numeric_limits<T>::min();
    for (const auto& s : globAlignment.shifts) {
        minX = std::min(std::floor(s.x), minX);
        maxX = std::max(std::ceil(s.x), maxX);
        minY = std::min(std::floor(s.y), minY);
        maxY = std::max(std::ceil(s.y), maxY);
    }
    auto res = std::make_pair(std::abs(maxX - minX), std::abs(maxY - minY));
    if (verbose) {
        std::cout << "Movie borders: x=" << res.first << " y=" << res.second
                << std::endl;
    }
    return res;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeBSplineCoeffs(const Dimensions &movieSize,
        const LocalAlignmentResult<T> &alignment,
        const Dimensions &controlPoints, const std::pair<size_t, size_t> &noOfPatches) {

    if(this->verbose) std::cout << "Computing BSpline coefficients" << std::endl;
    // get coefficients fo the BSpline that can represent the shifts (formula  from the paper)
    int lX = controlPoints.x();
    int lY = controlPoints.y();
    int lT = controlPoints.n();
    int noOfPatchesXY = noOfPatches.first * noOfPatches.second;
    Matrix2D<T>A(noOfPatchesXY*movieSize.n(), lX * lY * lT);
    Matrix1D<T>bX(noOfPatchesXY*movieSize.n());
    Matrix1D<T>bY(noOfPatchesXY*movieSize.n());
    T hX = (lX == 3) ? movieSize.x() : (movieSize.x() / (T)(lX-3));
    T hY = (lY == 3) ? movieSize.y() : (movieSize.y() / (T)(lY-3));
    T hT = (lT == 3) ? movieSize.n() : (movieSize.n() / (T)(lT-3));

    for (auto &&r : alignment.shifts) {
        auto meta = r.first;
        auto shift = r.second;
        int tileIdxT = meta.id_t;
        int tileCenterT = tileIdxT * 1 + 0 + 0;
        int tileIdxX = meta.id_x;
        int tileIdxY = meta.id_y;
        int tileCenterX = meta.rec.getCenter().x;
        int tileCenterY = meta.rec.getCenter().y;
        int i = (tileIdxY * noOfPatches.first) + tileIdxX;

        for (int j = 0; j < (lT * lY * lX); ++j) {
            int controlIdxT = (j / (lY * lX)) - 1;
            int XY = j % (lY * lX);
            int controlIdxY = (XY / lX) -1;
            int controlIdxX = (XY % lX) -1;
            // note: if control point is not in the tile vicinity, val == 0 and can be skipped
            T val = Bspline03((tileCenterX / (T)hX) - controlIdxX) *
                    Bspline03((tileCenterY / (T)hY) - controlIdxY) *
                    Bspline03((tileCenterT / (T)hT) - controlIdxT);
            MAT_ELEM(A,tileIdxT*noOfPatchesXY + i,j) = val;
        }
        VEC_ELEM(bX,tileIdxT*noOfPatchesXY + i) = -shift.x; // we want the BSPline describing opposite transformation,
        VEC_ELEM(bY,tileIdxT*noOfPatchesXY + i) = -shift.y; // so that we can use it to compensate for the shift
    }

    // solve the equation system for the spline coefficients
    Matrix1D<T> coefsX, coefsY;
    this->solveEquationSystem(bX, bY, A, coefsX, coefsY);
    return std::make_pair(coefsX, coefsY);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getLocalAlignmentCorrelationDownscale(
        const Dimensions &patchDim, T maxShift) {
    T minX = ((maxShift * 2) + 1) / patchDim.x();
    T minY = ((maxShift * 2) + 1) / patchDim.y();
    return std::make_pair(std::max(minX, (T)0.25), std::max(minY, (T)0.25));
}

template<typename T>
LocalAlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeLocalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &gain,
        const AlignmentResult<T> &globAlignment) {
    auto movieSettings = this->getMovieSettings(movie, false);
    auto patchSettings = this->getPatchSettings(movieSettings);
    auto correlationSettings = this->getCorrelationSettings(patchSettings,
            getLocalAlignmentCorrelationDownscale(patchSettings.dim, this->maxShift));
    auto borders = getMovieBorders(globAlignment, this->verbose);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings.dim,
            patchSettings.dim);
    if (this->verbose) {
        std::cout << "Settings for the patches: " << patchSettings << std::endl;
    }
    if (this->verbose) {
        std::cout << "Settings for the patches: " << correlationSettings << std::endl;
    }

    // load movie to memory
    if (nullptr == movieRawData) {
        movieRawData = loadMovie(movie, movieSettings, dark, gain);
    }
    T* movieData = movieRawData;
    movieRawData = nullptr; // currently, raw movie data are needed after local alignment

    // prepare filter
    MultidimArray<T> filter = this->createLPF(this->getTargetOccupancy(), correlationSettings.dim.x(),
            correlationSettings.dim.y());

    // compute max of frames in buffer
    T corrSizeMB = ((size_t) correlationSettings.x_freq
            * correlationSettings.dim.y()
            * sizeof(std::complex<T>))
            / ((T) 1024 * 1024);
    size_t framesInBuffer = std::ceil((gpu.value().lastFreeMem() / 3) / corrSizeMB);

    // prepare result
    LocalAlignmentResult<T> result { .globalHint = globAlignment };
    result.shifts.reserve(patchesLocation.size() * movieSettings.dim.n());
    auto refFrame = core::optional<size_t>(globAlignment.refFrame);

    // allocate additional memory for the patches
    size_t patchesElements = std::max(correlationSettings.elemsFreq(), correlationSettings.elemsSpacial());
    T *patchesData1 = new T[patchesElements];
    T *patchesData2 = new T[patchesElements];

    std::vector<std::thread> threads;

    // use additional thread that would load the data at the background
    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        // get data
        memset(patchesData1, 0, patchesElements * sizeof(T));
        getPatchData(movieData, p.rec, globAlignment, movieSettings.dim,
                patchesData1);
        // don't swap buffers while some thread is accessing its content
        alignDataMutex.lock();
        // swap buffers
        auto tmp = patchesData2;
        patchesData2 = patchesData1;
        patchesData1 = tmp;
        // run processing thread on the background
        threads.push_back(std::thread([&]() {
            // delay locking, but not for too long
            alignDataMutex.try_lock_for(std::chrono::seconds(1));
            std::cout << "\nProcessing patch " << p.id_x << " " << p.id_y << std::endl;
            // get alignment
            auto alignment = align(patchesData2, patchSettings,
                    correlationSettings, filter, refFrame,
                    this->maxShift, framesInBuffer, false);
            // process it
            for (size_t i = 0;i < movieSettings.dim.n();++i) {
                FramePatchMeta<T> tmp = p;
                // keep consistent with data loading
                int globShiftX = std::round(globAlignment.shifts.at(i).x);
                int globShiftY = std::round(globAlignment.shifts.at(i).y);
                tmp.id_t = i;
                // total shift is global shift + local shift
                result.shifts.emplace_back(tmp, Point2D<T>(globShiftX, globShiftY)
                        + alignment.shifts.at(i));
            }
        }));
        // thread should be created now, let it work and load new buffer meanwhile
        alignDataMutex.unlock();
    }
    // wait for the last processing thread
    for (auto &e : threads) {
        e.join();
    }

    delete[] movieData;
    delete[] patchesData1;
    delete[] patchesData2;
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::localFromGlobal(
        const MetaData& movie,
        const AlignmentResult<T> &globAlignment) {
    auto movieSettings = getMovieSettings(movie, false);
    LocalAlignmentResult<T> result { .globalHint = globAlignment };
    auto patchSettings = this->getPatchSettings(movieSettings);
    auto borders = getMovieBorders(globAlignment);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings.dim,
            patchSettings.dim);
    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        // process it
        for (size_t i = 0; i < movieSettings.dim.n(); ++i) {
            FramePatchMeta<T> tmp = p;
            tmp.id_t = i;
            result.shifts.emplace_back(tmp, Point2D<T>(globAlignment.shifts.at(i).x, globAlignment.shifts.at(i).y));
        }
    }
    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const AlignmentResult<T> &globAlignment) {
    applyShiftsComputeAverage(movie, dark, gain, initialMic, Ninitial, averageMicrograph,
            N, localFromGlobal(movie, globAlignment));
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeCoefficients(std::pair<Matrix1D<T>, Matrix1D<T>> &coeffs) {
    if (fnBSplinePath.isEmpty()) return;

    auto fnX = FileName(fnBSplinePath.getBaseName() + "X", fnBSplinePath.getExtension());
    auto fnY = FileName(fnBSplinePath.getBaseName() + "Y", fnBSplinePath.getExtension());

    coeffs.first.write(fnX);
    coeffs.second.write(fnY);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const LocalAlignmentResult<T> &alignment) {
    // Apply shifts and compute average
    Image<T> croppedFrame, reducedFrame, shiftedFrame;
    int j = 0;
    int n = 0;
    Ninitial = N = 0;
    GeoTransformer<T> transformer;
    auto movieSettings = getMovieSettings(movie, false);

    auto coeffs = computeBSplineCoeffs(movieSettings.dim, alignment, localAlignmentControlPoints, localAlignPatches);
    storeCoefficients(coeffs);


    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        if (n >= this->nfirstSum && n <= this->nlastSum) {
            // load frame
            this->loadFrame(movie, dark, gain, __iter.objId, croppedFrame);

            if (this->bin > 0) {
                // FIXME add templates to respective functions/classes to avoid type casting
                Image<double> croppedFrameDouble;
                Image<double> reducedFrameDouble;
                typeCast(croppedFrame(), croppedFrameDouble());

                scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / this->bin),
                        floor(XSIZE(croppedFrame()) / this->bin),
                        croppedFrameDouble(), reducedFrameDouble());

                typeCast(reducedFrameDouble(), reducedFrame());

                croppedFrame() = reducedFrame();
            }

            if ( ! this->fnInitialAvg.isEmpty()) {
                if (j == 0)
                    initialMic() = croppedFrame();
                else
                    initialMic() += croppedFrame();
                Ninitial++;
            }

            if (this->fnAligned != "" || this->fnAvg != "") {
                transformer.initLazyForBSpline(croppedFrame.data.xdim, croppedFrame.data.ydim, movieSettings.dim.n(),
                    localAlignmentControlPoints.x(), localAlignmentControlPoints.y(), localAlignmentControlPoints.n());
                transformer.applyBSplineTransform(this->BsplineOrder, shiftedFrame(), croppedFrame(), coeffs, j);


                if (this->fnAligned != "")
                    shiftedFrame.write(this->fnAligned, j + 1, true,
                            WRITE_REPLACE);
                if (this->fnAvg != "") {
                    if (j == 0)
                        averageMicrograph() = shiftedFrame();
                    else
                        averageMicrograph() += shiftedFrame();
                    N++;
                }
            }
            std::cout << "Frame " << std::to_string(j) << " processed." << std::endl;
            j++;
        }
        n++;
    }
}

template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeGlobalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &gain) {
    auto movieSettings = this->getMovieSettings(movie, true);
    T sizeFactor = this->computeSizeFactor();
    if (this->verbose) {
        std::cout << "Settings for the movie: " << movieSettings << std::endl;
    }
    auto correlationSetting = this->getCorrelationSettings(movieSettings,
            std::make_pair(sizeFactor, sizeFactor));
    if (this->verbose) {
        std::cout << "Settings for the correlation: " << correlationSetting << std::endl;
    }

    MultidimArray<T> filter = this->createLPF(this->getTargetOccupancy(), correlationSetting.dim.x(),
            correlationSetting.dim.y());

    T corrSizeMB = ((size_t) correlationSetting.x_freq
            * correlationSetting.dim.y()
            * sizeof(std::complex<T>)) / ((T) 1024 * 1024);
    size_t framesInBuffer = std::ceil((gpu.value().lastFreeMem() / 3) / corrSizeMB);

    auto reference = core::optional<size_t>();


    // load movie to memory
    if (nullptr == movieRawData) {
        movieRawData = loadMovie(movie, movieSettings, dark, gain);
    }
    size_t elems = std::max(movieSettings.elemsFreq(), movieSettings.elemsSpacial());
    T* data = new T[elems];
    memcpy(data, movieRawData, elems * sizeof(T));

    // lock the data processing (as alignment will unlock it)
    alignDataMutex.lock();
    auto result = align(data, movieSettings, correlationSetting,
                    filter, reference,
            this->maxShift, framesInBuffer, this->verbose);
    delete[] data;
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::align(T *data,
        const FFTSettings<T> &in, const FFTSettings<T> &correlation,
        MultidimArray<T> &filter,
        core::optional<size_t> &refFrame,
        size_t maxShift, size_t framesInCorrelationBuffer, bool verbose) {
    assert(nullptr != data);
    size_t N = in.dim.n();
    // scale and transform to FFT on GPU
    performFFTAndScale<T>(data, N, in.dim.x(), in.dim.y(), in.batch,
            correlation.x_freq, correlation.dim.y(), filter);

    auto scale = std::make_pair(in.dim.x() / (T) correlation.dim.x(),
            in.dim.y() / (T) correlation.dim.y());

    return computeShifts(verbose, maxShift, (std::complex<T>*) data, correlation,
            in.dim.n(),
            scale, framesInCorrelationBuffer, refFrame);
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadMovie(const MetaData& movie,
        const FFTSettings<T> &settings, const Image<T>& dark,
        const Image<T>& gain) {
    // allocate enough memory for the images. Since it will be reused, it has to be big
    // enough to store either all FFTs or all input images
    T* imgs = new T[std::max(settings.elemsFreq(), settings.elemsSpacial())]();
    Image<T> frame;

    int movieImgIndex = -1;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        // update variables
        movieImgIndex++;
        if (movieImgIndex < this->nfirst) continue;
        if (movieImgIndex > this->nlast) break;

        // load image
        this->loadFrame(movie, dark, gain, __iter.objId, frame);

        // copy line by line, adding offset at the end of each line
        // result is the same image, padded in the X and Y dimensions
        T* dest = imgs
                + ((movieImgIndex - this->nfirst) * settings.dim.x()
                        * settings.dim.y()); // points to first float in the image
        for (size_t i = 0; i < settings.dim.y(); ++i) {
            memcpy(dest + (settings.dim.x() * i),
                    frame.data.data + i * frame.data.xdim,
                    settings.dim.x() * sizeof(T));
        }
    }
    return imgs;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeShifts(bool verbose,
        size_t maxShift,
        std::complex<T>* data, const FFTSettings<T>& settings, size_t N,
        std::pair<T, T>& scale,
        size_t framesInCorrelationBuffer,
        const core::optional<size_t>& refFrame) {
    // N is number of images, n is number of correlations
    // compute correlations (each frame with following ones)
    T* correlations;
    size_t centerSize = std::ceil(maxShift * 2 + 1);
    computeCorrelations(centerSize, N, data, settings.x_freq,
            settings.dim.x(),
            settings.dim.y(), framesInCorrelationBuffer,
            settings.batch, correlations);
    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift

    // we are done with the input data, so release it
    alignDataMutex.unlock();
    Matrix2D<T> A(N * (N - 1) / 2, N - 1);
    Matrix1D<T> bX(N * (N - 1) / 2), bY(N * (N - 1) / 2);

    // find the actual shift (max peak) for each pair of frames
    // and create a set or equations
    size_t idx = 0;
    MultidimArray<T> Mcorr(centerSize, centerSize);
    T* origData = Mcorr.data;

    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            size_t offset = idx * centerSize * centerSize;
            Mcorr.data = correlations + offset;
            Mcorr.setXmippOrigin();
            bestShift(Mcorr, bX(idx), bY(idx), NULL,
                    maxShift / scale.first);
            bX(idx) *= scale.first; // scale to expected size
            bY(idx) *= scale.second;
            if (verbose) {
                std::cerr << "Frame " << i << " to Frame " << j << " -> ("
                        << bX(idx) << "," << bY(idx) << ")" << std::endl;
            }
            for (int ij = i; ij < j; ij++) {
                A(idx, ij) = 1;
            }
            idx++;
        }
    }
    Mcorr.data = origData;
    delete[] correlations;

    // now get the estimated shift (from the equation system)
    // from each frame to successing frame
    AlignmentResult<T> result = this->computeAlignment(bX, bY, A, refFrame, N);
    return result;
}

template<typename T>
int ProgMovieAlignmentCorrelationGPU<T>::getMaxFilterSize(
        const Image<T> &frame) {
    size_t maxXPow2 = std::ceil(log(frame.data.xdim) / log(2));
    size_t maxX = std::pow(2, maxXPow2);
    size_t maxFFTX = maxX / 2 + 1;
    size_t maxYPow2 = std::ceil(log(frame.data.ydim) / log(2));
    size_t maxY = std::pow(2, maxYPow2);
    size_t bytes = maxFFTX * maxY * sizeof(T);
    return bytes / (1024 * 1024);
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
