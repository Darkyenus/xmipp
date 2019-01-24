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
import subprocess

import sys
import os
import shutil
from os.path import dirname, realpath, join, isfile, exists

VERSION_TAG = 'Xmipp version'
VERSION_CMD = "build/bin/xmipp_version"

def usage(error):
    print ("\n"
           "    ERROR: %s\n"
           "\n"
           "    Usage: python tar.py <mode> [version]\n"
           "\n"
           "             mode: Binaries: Just the binaries \n"
           "                   Sources: Just the source code.\n"
           "\n"
           "             version: X.YY.MM  (version, year and month)\n"
           "                      the default version is taken from %s\n"
           "    ") % (error, VERSION_CMD)
    sys.exit(1)

def getVersion():
    try:
        result = subprocess.check_output(VERSION_CMD).splitlines()
    except:
        raise Exception("The '%s' file (containing the '%s') do NOT exist."
                        % (VERSION_CMD, VERSION_TAG))
    for line in result:

        if line.startswith(VERSION_TAG):
            print(line)
            return line.split(": ")[1]
    raise Exception("'xmipp_version' is not returning the '%s' information"
                    % VERSION_TAG)

def run(*argv):
    if len(argv) == 1:
        label = argv[0]
        version = getVersion()
    elif len(argv) == 2:
        label = argv[0]
        version = argv[1]
    else:
        print("Incorrect number of parameters")
        return

    XMIPP_PATH = realpath(dirname(dirname(dirname(realpath(__file__)))))
    MODES = {'Binaries': 'build', 'Sources': 'src'}

    def makeTarget(target, label):
        if exists(target):
            print("%s already exists. Removing it...")
            os.system("rm -rf %s" % target)
        print("...preparing the bundle...")
        shutil.copytree(MODES[label], target, symlinks=True)

    excludeTgz = ''
    tgzPath = "xmipp%s-%s"
    if label == 'Binaries':
        print("Recompiling to make sure that last version is there...")
        try:
            # doing compilation and install separately to skip config
            os.system("./xmipp compile 4")
            os.system("./xmipp install")
        except:
            print("  ...some error occurred during the compilation.\nFollowing with the bundle creation.")
        target = tgzPath % ('Bin', version)
        if not isfile(join(XMIPP_PATH, 'build', 'bin', 'xmipp_reconstruct_significant')):
            print("\n"
                  "     ERROR: %s not found. \n"
                  "            Xmipp needs to be compiled to make the binaries.tgz."
                  % target)
            sys.exit(1)
        excludeTgz = "--exclude='*.tgz' --exclude='*.h' --exclude='*.cpp' --exclude='*.java'"
        makeTarget(target, label)
    elif label == 'Sources':
        target = tgzPath % ('Src', version)
        os.mkdir(target)
        makeTarget(join(target, 'src'), label)
    else:
        usage("Incorrect <mode>")


    args = {'excludeTgz': excludeTgz,
            'target': target}

    cmdStr = "tar czf %(target)s.tgz --exclude=.git --exclude='software/tmp/*' " \
             "--exclude='*.o' --exclude='*.os' --exclude='*pyc' " \
             "--exclude='*.mrc' --exclude='*.stk' --exclude='*.gz' %(excludeTgz)s" \
             "--exclude='*.scons*' --exclude='config/*.conf' %(target)s"

    cmd = cmdStr % args

    if exists(target+'.tgz'):
        print("%s.tgz already exists. Removing it...")
        os.system("rm -rf %s.tgz" % target)

    print(cmd)
    os.system(cmd)
    os.system("rm -rf %s" % target)


if __name__  == '__main__':

    if not (len(sys.argv) == 2 or len(sys.argv) == 3):
        usage("Incorrect number of input parameters")

    label = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) == 3 else getVersion()

    run(label, version)
