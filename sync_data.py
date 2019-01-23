#!/usr/bin/env python
# ***************************************************************************
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/

import hashlib
import os
import sys

from subprocess import call
from os.path import join
from urllib2 import urlopen

import time


def download(destination=None, url=None, dataset=None, isDLmodel=False):
    """ Download all the data files mentioned in url/dataset/MANIFEST
    """
    if not isDLmodel:
        # First make sure that we ask for a known dataset.
        if dataset not in [x.strip('./\n') for x in urlopen('%s/MANIFEST'%url)]:
            print "Unknown dataset/model: %s" % dataset
            return
        remoteManifest = '%s/%s/MANIFEST' % (url, dataset)
        inFolder = "/%s" % dataset
    else:
        remoteManifest = '%s/xmipp_models_MANIFEST' % url
        inFolder = ''

    # Retrieve the dataset's MANIFEST file.
    # It contains a list of "file md5sum" of all files included in the dataset.
    os.makedirs(destination)
    manifest = join(destination, 'MANIFEST')
    try:
        print "Retrieving MANIFEST file"
        open(manifest, 'w').writelines(
            urlopen(remoteManifest))
    except Exception as e:
        sys.exit("ERROR reading %s (%s)" % (remoteManifest, e))

    # Now retrieve all of the files mentioned in MANIFEST, and check their md5.
    print 'Fetching files...'
    md5sRemote = readManifest(remoteManifest, isDLmodel)
    done = 0.0  # fraction already done
    inc = 1.0 / len(md5sRemote)  # increment, how much each iteration represents
    oldPartial = 100
    for fname, md5Remote in md5sRemote.iteritems():
        fpath = join(destination, fname)
        try:
            # Download content and create file with it.
            if not os.path.isdir(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
            open(fpath, 'w').writelines(
                urlopen('%s%s/%s' % (url, inFolder, fname)))

            md5 = md5sum(fpath)
            assert md5 == md5Remote, \
                "Bad md5. Expected: %s Computed: %s" % (md5Remote, md5)

            done += inc
            partial = int(done * 10)
            if int(done * 100 % 10) == 0 and partial != oldPartial:
                sys.stdout.write("%3d%%..." % (100 * done))
                sys.stdout.flush()
                oldPartial = partial
        except Exception as e:
            print "\nError in %s (%s)" % (fname, e)
            print "URL: %s/%s/%s" % (url, dataset, fname)
            print "Destination: %s" % fpath
            if raw_input("Continue downloading? (y/[n]): ").lower() != 'y':
                sys.exit()
    print
    return md5sRemote.keys()

def update(destination=None, url=None, dataset=None, isDLmodel=False):
    """ Update local dataset with the contents of the remote one.
    It compares the md5 of remote files in url/dataset/MANIFEST with the
    ones in workingCopy/dataset/MANIFEST, and downloads only when necessary.
    """
    prefix = "xmipp_models_" if isDLmodel else ''
    inFolder = "" if isDLmodel else "/%s" % dataset

    # Read contents of *remote* MANIFEST file, and create a dict {fname: md5}
    remoteManifest = '%s/%sMANIFEST' % (url, prefix) if isDLmodel \
                         else '%s/%s/MANIFEST' % (url, dataset)

    md5sRemote = readManifest(remoteManifest, isDLmodel)

    # Update and read contents of *local* MANIFEST file, and create a dict
    try:
        last = max(os.stat(join(destination, x)).st_mtime for x in md5sRemote)
        t_manifest = os.stat(join(destination, 'MANIFEST')).st_mtime
        assert t_manifest > last and time.time() - t_manifest < 60*60*24*7
    except (OSError, IOError, AssertionError) as e:
        print "Regenerating local MANIFEST..."
        if isDLmodel:
            os.system('(cd %s ; md5sum xmipp_model_*.tgz '
                      '> MANIFEST)' % destination)
        else:
            createMANIFEST(destination)
    md5sLocal = dict(x.split() for x in open(join(destination, 'MANIFEST')))
    if isDLmodel:  # DLmodels has hashs before fileNames
        md5sLocal = {v: k for k, v in md5sLocal.iteritems()}

    # Check that all the files mentioned in MANIFEST are up-to-date
    print "Verifying MD5s..."

    filesUpdated = []  # number of files that have been updated
    taintedMANIFEST = False  # can MANIFEST be out of sync?

    done = 0.0  # fraction already done
    inc = 1.0 / len(md5sRemote)  # increment, how much each iteration represents
    oldPartial = 100
    for fname in md5sRemote:
        fpath = join(destination, fname)
        # try:
        if os.path.exists(fpath) and md5sLocal[fname] == md5sRemote[fname]:
            pass  # just to emphasize that we do nothing in this case
        else:
            if not os.path.isdir(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
            open(fpath, 'w').writelines(
                urlopen('%s%s/%s' % (url, inFolder, fname)))
            filesUpdated.append(fname)
        # except Exception as e:
        #     print "\nError while updating %s: %s" % (fname, e)
        #     taintedMANIFEST = True  # if we don't update, it can be wrong
        done += inc
        partial = int(done*10)
        if int(done*100%10) == 0 and partial != oldPartial:
                sys.stdout.write("%3d%%..." % (100 * done))
                sys.stdout.flush()
                oldPartial = partial

    sys.stdout.write("\n...done. Updated files: %d\n" % len(filesUpdated))
    sys.stdout.flush()

    # Save the new MANIFEST file in the folder of the downloaded dataset
    if len(filesUpdated) > 0:
        open(join(destination, 'MANIFEST'), 'w').writelines(manifest)

    if taintedMANIFEST:
        print "Some files could not be updated. Regenerating local MANIFEST ..."
        createMANIFEST(destination)

    return md5sRemote.keys()


def upload(login, localFn, remoteFolder, isDLmodel=False):
    """ Upload a dataset to our repository
    """
    if not os.path.exists(localFn):
        sys.exit("ERROR: local folder/file %s does not exist." % localFn)

    print "Warning: Uploading, please BE CAREFUL! This can be dangerous."
    print ('You are going to be connected to "%s" to write in folder '
           '"%s".' % (login, remoteFolder))
    if raw_input("Continue? YES/no").lower() == 'no':
        sys.exit()

    # Upload the dataset files (with rsync)
    try:
        print "Uploading files..."
        call(['rsync', '-rlv', '--chmod=a+r', localFn,
              '%s:%s' % (login, remoteFolder)])
    except:
        sys.exit("Uload failed, you may have a permissions issues.\n"
                 "Please check the login introduced or contact to "
                 "'scipion@cnb.csic.es' to upload the model.")

    # Regenerate remote MANIFEST (which contains a list of datasets)
    if isDLmodel:
        # This is a file that just contains the name of the xmipp_models
        # in remoteFolder. Nothing to do with the MANIFEST files in
        # the datasets, which contain file names and md5s.
        print "Regenerating remote MANIFEST file..."
        call(['ssh', login,
              'cd %s && md5sum xmipp_model_*.tgz > xmipp_models_MANIFEST'
              % remoteFolder])

def md5sum(fname):
    """ Return the md5 hash of file fname
    """
    mhash = hashlib.md5()
    with open(fname) as f:
        for chunk in iter(lambda: f.read(128 * mhash.block_size), ''):
            mhash.update(chunk)
    return mhash.hexdigest()

def createMANIFEST(path):
    """ Create a MANIFEST file in path with the md5 of all files below
    """
    with open(join(path, 'MANIFEST'), 'w') as manifest:
        for root, dirs, files in os.walk(path):
            for filename in set(files) - {'MANIFEST'}:  # all but ourselves
                fn = join(root, filename)  # file to check
                manifest.write('%s %s\n' % (os.path.relpath(fn, path), md5sum(fn)))

def readManifest(remoteManifest, isDLmodel):
    manifest = urlopen(remoteManifest).readlines()
    md5sRemote = dict(x.strip().split() for x in manifest)
    if isDLmodel:  # DLmodels has hashs before fileNames
        md5sRemote = {v: k for k, v in md5sRemote.iteritems()}
    return md5sRemote
