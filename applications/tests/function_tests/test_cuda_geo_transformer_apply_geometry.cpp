#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "core/transformations.h"
#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"

template <typename A, int B>
struct TypeDefinitions
{
	using ScalarType = A;
	static const int SplineDegree = B;
};

template< typename T >
class GeoTransformerApplyGeometryTest : public ::testing::Test {
public:

	using ScalarType = typename T::ScalarType;
	static const int SplineDegree = T::SplineDegree;

    void compare_results( double* true_values, double* approx_values ) {
        for ( int i = 0; i < y * x; ++i ) {
            ASSERT_NEAR( true_values[i], approx_values[i], 1e-12 ) << "at index:" << i;
        }
    }

    /*
     * For big matrices there can be a small number of pixels whose value differs more than 0.0001
     * from reference values, therefore I test it for lower precision with floats
     * If tested with double values we require high precision
    */
    void compare_results( float* true_values, float* approx_values ) {
        for ( int i = 0; i < y * x; ++i ) {
            ASSERT_NEAR( true_values[i], approx_values[i], 1e-3f ) << "at index:" << i;
        }
    }

    void allocate_arrays() {
        in.resize( y, x );
        out.resize( y, x );
        out_ref.resize( y, x );
    }

    void run_transformation() {
        GeoTransformer< ScalarType > gt;
        gt.initForMatrix(in.xdim, in.ydim, in.zdim);
        gt.applyGeometry( SplineDegree, out, in, transform, false, true );
    }

    void compute_reference_result() {
        ::applyGeometry( SplineDegree, out_ref, in, transform, false, true );
    }

    std::pair< size_t, size_t > random_size( int seed ) {
        gen.seed( seed );
        std::uniform_int_distribution<> dis( 128, 8192 );

        return { dis( gen ), dis( gen ) };
    }

    void randomly_initialize( MultidimArray< ScalarType >& array, int seed ) {
        gen.seed( seed );
        std::uniform_real_distribution<> dis( -1.0, 1.0 );

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY( array ) {
            DIRECT_MULTIDIM_ELEM( array, n ) = dis( gen );
        }
    }

    void randomly_initialize(Matrix1D< ScalarType >& array, int seed) {
        gen.seed( seed );
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        FOR_ALL_ELEMENTS_IN_MATRIX1D( array ) {
            MATRIX1D_ARRAY( array )[i] = dis(gen);
        }

    }

    void init_transform(int seed) {
    	gen.seed( seed );
        std::uniform_real_distribution<> dis( -20.0, 20.0 );
        init_transform( dis( gen ), dis( gen ) );
    }

    void init_transform(float x_shift, float y_shift) {
    	Matrix1D< ScalarType > shift(2);
	    shift.vdata[0] = x_shift;
	    shift.vdata[1] = y_shift;
	    translation2DMatrix(shift, transform, true);
    }

    size_t x;
    size_t y;

    ScalarType outside = 0;

    Matrix2D< ScalarType > transform;

    MultidimArray< ScalarType > in;
    MultidimArray< ScalarType > out;
    MultidimArray< ScalarType > out_ref;

    std::mt19937 gen;
};

TYPED_TEST_CASE_P(GeoTransformerApplyGeometryTest);

TYPED_TEST_P(GeoTransformerApplyGeometryTest, OriginalTestFromGeoTransformer) {
    this->x = 64;
    this->y = 64;
    this->allocate_arrays();
    this->init_transform( 0.45, 0.62 );

    for (int i = 0; i < this->in.ydim; ++i) {
        for (int j = 0; j < this->in.xdim; ++j) {
            this->in.data[i * this->in.xdim + j] = i * 10 + j;
        }
    }

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out_ref.data, this->out.data );
}


TYPED_TEST_P(GeoTransformerApplyGeometryTest, ZeroStaysZero) {
    this->x = 256;
    this->y = 256;
    this->allocate_arrays();
    this->init_transform( 5 );

    this->run_transformation();

    this->compare_results( this->in.data, this->out.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, NoChangeIfShiftIsZero) {
    this->x = 337;
    this->y = 240;
    this->allocate_arrays();
    this->init_transform( 0, 0 );

    this->randomly_initialize( this->in, 13 );

    this->run_transformation();

    this->compare_results( this->in.data, this->out.data );

}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, ZeroInputNonzeroTransformIsZeroOutput) {
    this->x = 169;
    this->y = 169;
    this->allocate_arrays();
    this->init_transform( 7 );

    this->run_transformation();

    this->compare_results( this->in.data, this->out.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, RandomInputWithNonzeroTransform) {
    this->x = 256;
    this->y = 128;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 23 );
    this->init_transform( 13 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, EvenButNotPaddedInput) {
    this->x = 332;
    this->y = 420;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 24 );
    this->init_transform( 11 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, OddEvenSizedInput) {
    this->x = 321;
    this->y = 168;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 63 );
    this->init_transform( 17 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, EvenOddSizedInput) {
    this->x = 258;
    this->y = 203;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 10 );
    this->init_transform( 19 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, BiggerSize4K) {
    this->x = 3840;
    this->y = 2160;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 11 );
    this->init_transform( 23 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data );
}

TYPED_TEST_P(GeoTransformerApplyGeometryTest, BiggerSizeInOneDimension) {
    this->x = 3840;
    this->y = 256;
    this->allocate_arrays();
    this->randomly_initialize( this->in, 81 );
    this->init_transform( 31 );

    this->run_transformation();
    this->compute_reference_result();

    this->compare_results( this->out.data, this->out_ref.data );
}

REGISTER_TYPED_TEST_CASE_P(GeoTransformerApplyGeometryTest,
	OriginalTestFromGeoTransformer,
    ZeroStaysZero,
    NoChangeIfShiftIsZero,
    ZeroInputNonzeroTransformIsZeroOutput,
    RandomInputWithNonzeroTransform,
    EvenButNotPaddedInput,
    OddEvenSizedInput,
    EvenOddSizedInput,
    BiggerSize4K,
    BiggerSizeInOneDimension
);

using TestTypes = ::testing::Types< TypeDefinitions<float, 3>, TypeDefinitions<double, 3>/*,
                    TypeDefinitions<float, 1>, TypeDefinitions<double, 1>*/ >;
INSTANTIATE_TYPED_TEST_CASE_P(TestTypesInstantiation, GeoTransformerApplyGeometryTest, TestTypes);

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}