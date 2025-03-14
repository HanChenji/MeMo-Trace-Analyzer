import os, sys
from os.path import join as joinpath

useIcc = False
#useIcc = True

libhdf5_link = 'https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz'
libconfig_link = 'https://github.com/hyperrealm/libconfig/archive/refs/tags/v1.7.3.tar.gz'
libelf_link = 'https://codeload.github.com/WolfgangSt/libelf/zip/refs/heads/master'
pin_link = 'https://software.intel.com/sites/landingpage/pintool/downloads/pin-2.14-71313-gcc.4.4.7-linux.tar.gz'

def check_libconf(dir):
    if not os.path.exists("libs/libconfig"):
        os.system(f"mkdir -p libs && cd libs && wget {libconfig_link} && tar xf v1.7.3.tar.gz && mv libconfig-1.7.3 libconfig && rm v1.7.3.tar.gz")
    if not os.path.exists("libs/libconfig/build/lib/libconfig++.so"):
        os.system(f"cd libs/libconfig && autoreconf && ./configure --prefix={dir} && make -j ; make install")
    print("libconf is ready")

def check_libhdf5(dir):
    if not os.path.exists("libs/libhdf5"):
        os.system(f"mkdir -p libs && cd libs && wget {libhdf5_link} && tar xf hdf5-1_14_3.tar.gz && mv hdf5-hdf5-1_14_3 libhdf5 && rm hdf5-1_14_3.tar.gz")
    if not os.path.exists("libs/libhdf5/build/lib/libhdf5.so"):
        os.system(f"cd libs/libhdf5 && ./configure --prefix={dir} && make -j && make install")
    print("libhdf5 is ready")

def check_libelf(dir):
    if not os.path.exists("libs/libelf"):
        os.system(f"mkdir -p libs && cd libs && wget {libelf_link} && unzip master && mv libelf-master libelf && rm master")
    if not os.path.exists("libs/libelf/build/lib/libelf.so"):
        os.system(f"cd libs/libelf && ./configure --prefix={dir} && make -j && make install")
    print("libelf is ready")

def check_pin():
    if not os.path.exists("pin-2.14-71313-gcc.4.4.7-linux"):
        os.system(f"wget {pin_link} && tar xf pin-2.14-71313-gcc.4.4.7-linux.tar.gz && rm pin-2.14-71313-gcc.4.4.7-linux.tar.gz")
    print("pin is ready")

def git_ver():
    import os
    def cmd(c): return os.popen(c).read().strip()
    branch = cmd("git rev-parse --abbrev-ref HEAD")
    revnum = cmd("git log | grep ^commit | wc -l")
    rshort = cmd("git rev-parse --short HEAD")
    dfstat = cmd("git diff HEAD --shortstat")
    dfhash = cmd("git diff HEAD | md5sum")[:8]
    shstat = dfstat.replace(" files changed", "fc").replace(" file changed", "fc") \
                .replace(" insertions(+)", "+").replace(" insertion(+)", "+") \
                .replace(" deletions(-)", "-").replace(" deletion(-)", "-") \
                .replace(",", "") 
    diff = "clean" if len(dfstat) == 0 else shstat +  " " + dfhash
    return ":".join([branch, revnum, rshort, diff])

def buildSim(cppFlags, dir, type, pgo=None):
    ''' Build the simulator with a specific base buid dir and config type'''

    ROOT = Dir('.').abspath

    buildDir = joinpath(dir, type)
    print("Building " + type + " zsim at " + buildDir)

    env = Environment(ENV = os.environ, tools = ['default', 'textfile'])
    env["CPPFLAGS"] = cppFlags

    allSrcs = [f for dir, subdirs, files in os.walk("src") for f in Glob(dir + "/*")]
    versionFile = joinpath(buildDir, "version.h")
    if os.path.exists(".git"):
        env.Command(versionFile, allSrcs + [".git/index", "SConstruct"],
            f'printf "#define ZSIM_BUILDDATE \\"`date`\\"\\n#define ZSIM_BUILDVERSION \\"{git_ver()}\\"" >>' + versionFile)
    else:
        env.Command(versionFile, allSrcs + ["SConstruct"],
            'printf "#define ZSIM_BUILDDATE \\"`date`\\"\\n#define ZSIM_BUILDVERSION \\"no git repo\\"" >>' + versionFile)

    libconfig_path = joinpath(ROOT, "libs/libconfig/build")
    check_libconf(libconfig_path)
    env.Append(ENV = {'LIBCONFIGPATH' : libconfig_path})

    libhdf5_path = joinpath(ROOT, "libs/libhdf5/build")
    check_libhdf5(libhdf5_path)
    env.Append(ENV = {'HDF5PATH' : libhdf5_path})

    libelf_path = joinpath(ROOT, "libs/libelf/build")
    check_libelf(libelf_path)
    env.Append(ENV = {'LIBELFPATH' : libelf_path})

    pin_path = joinpath(ROOT, "pin-2.14-71313-gcc.4.4.7-linux")
    check_pin()
    env.Append(ENV = {'PINPATH' : pin_path})

    # Parallel builds?
    #env.SetOption('num_jobs', 32)

    # Use link-time optimization? It's still a bit buggy, so be careful
    #env['CXX'] = 'g++ -flto -flto-report -fuse-linker-plugin'
    #env['CC'] = 'gcc -flto'
    #env["LINKFLAGS"] = " -O3 -finline "
    if useIcc:
        env['CC'] = 'icc'
        env['CXX'] = 'icpc -ipo'

    # Required paths
    if "PINPATH" in os.environ:
        PINPATH = os.environ["PINPATH"]
    else:
       print("ERROR: You need to define the $PINPATH environment variable with Pin's path")
       sys.exit(1)

    # NOTE: These flags are for the 28/02/2011 2.9 PIN kit (rev39599). Older versions will not build.
    # NOTE (dsm 10 Jan 2013): Tested with Pin 2.10 thru 2.12 as well
    # NOTE: Original Pin flags included -fno-strict-aliasing, but zsim does not do type punning
    # NOTE (dsm 16 Apr 2015): Update flags code to support Pin 2.14 while retaining backwards compatibility
    env["CPPFLAGS"] += " -std=c++0x -Wall -Wno-unknown-pragmas -Wno-deprecated -Wno-unused-function -Wno-catch-value -fomit-frame-pointer -fno-stack-protector"
    env["CPPFLAGS"] += " -DMALLOC_ALIGNMENT=64"
    env["CPPFLAGS"] += " -MMD -DBIGARRAY_MULTIPLIER=1 -DUSING_XED -DTARGET_IA32E -DHOST_IA32E -fPIC -DTARGET_LINUX"
    # NOTE (yyf 17 Sep 2020): Enforce to use older ABI when compiled with gcc-5 or newer.
    env["CPPFLAGS"] += " -fabi-version=2 -D_GLIBCXX_USE_CXX11_ABI=0"
    env["CPPFLAGS"] += " -Ofast"

    # Pin 2.12+ kits have changed the layout of includes, detect whether we need
    # source/include/ or source/include/pin/
    pinInclDir = joinpath(PINPATH, "source/include/")
    if not os.path.exists(joinpath(pinInclDir, "pin.H")):
        pinInclDir = joinpath(pinInclDir, "pin")
        assert os.path.exists(joinpath(pinInclDir, "pin.H"))

    # Pin 2.14 changes location of XED
    xedName = "xed2"  # used below
    xedPath = joinpath(PINPATH, "extras/" + xedName + "-intel64/include")
    if not os.path.exists(xedPath):
        xedName = "xed"
        xedPath = joinpath(PINPATH, "extras/" + xedName + "-intel64/include")
        assert os.path.exists(xedPath)

    env["CPPPATH"] = [xedPath,
            pinInclDir, joinpath(pinInclDir, "gen"),
            joinpath(PINPATH, "extras/components/include")]

    # Uncomment to get logging messages to stderr
    ##env["CPPFLAGS"] += " -DDEBUG=1"

    # Be a Warning Nazi? (recommended)
    env["CPPFLAGS"] += " -Werror "

    # Enables lib and harness to use the same info/log code,
    # but only lib uses pin locks for thread safety
    env["PINCPPFLAGS"] = " -DMT_SAFE_LOG "

    # PIN-specific libraries
    env["PINLINKFLAGS"] = " -Wl,--hash-style=sysv -Wl,-Bsymbolic -Wl,--version-script=" + joinpath(pinInclDir, "pintool.ver")

    # To prime system libs, we include /usr/lib and /usr/lib/x86_64-linux-gnu
    # first in lib path. In particular, this solves the issue that, in some
    # systems, Pin's libelf takes precedence over the system's, but it does not
    # include symbols that we need or it's a different variant (we need
    # libelfg0-dev in Ubuntu systems)
    env["PINLIBPATH"] = [
        joinpath(PINPATH, "extras/" + xedName + "-intel64/lib"),
        joinpath(PINPATH, "intel64/lib"),
        joinpath(PINPATH, "intel64/lib-ext")
    ]

    # Libdwarf is provided in static and shared variants, Ubuntu only provides
    # static, and I don't want to add -R<pin path/intel64/lib-ext> because
    # there are some other old libraries provided there (e.g., libelf) and I
    # want to use the system libs as much as possible. So link directly to the
    # static version of libdwarf.

    # Pin 2.14 uses unambiguous libpindwarf
    pindwarfPath = joinpath(PINPATH, "intel64/lib-ext/libdwarf.a")
    pindwarfLib = File(pindwarfPath)
    if not os.path.exists(pindwarfPath):
        pindwarfLib = "pindwarf"

    env["PINLIBS"] = ["pin", "xed", pindwarfLib, "elf", "dl", "rt"]

    # Non-pintool libraries
    env["LIBPATH"] = []
    env["LIBS"] = ["config++"]

    env["LINKFLAGS"] = ""
    env["RPATH"] = []

    # Use non-standard library paths if defined
    if "LIBELFPATH" in os.environ:
        LIBELFPATH = os.environ["LIBELFPATH"]
        env["CPPPATH"] += [joinpath(LIBELFPATH, "include")]
        env["LIBPATH"] += [joinpath(LIBELFPATH, "lib")]
        env["RPATH"]   += [joinpath(LIBELFPATH, "lib")]

    if "LIBCONFIGPATH" in os.environ:
        LIBCONFIGPATH = os.environ["LIBCONFIGPATH"]
        env["CPPPATH"] += [joinpath(LIBCONFIGPATH, "include")]
        env["LIBPATH"] += [joinpath(LIBCONFIGPATH, "lib")]
        env["RPATH"]   += [joinpath(LIBCONFIGPATH, "lib")]

    if "HDF5PATH" in os.environ:
        HDF5PATH = os.getenv("HDF5PATH")
        env["CPPPATH"] += [joinpath(HDF5PATH, "include")]
        env["LIBPATH"] += [joinpath(HDF5PATH, "lib")]
        env["RPATH"]   += [joinpath(HDF5PATH, "lib")]

    if "POLARSSLPATH" in os.environ:
        POLARSSLPATH = os.environ["POLARSSLPATH"]
        env["PINLIBPATH"] += [joinpath(POLARSSLPATH, "library")]
        env["CPPPATH"] += [joinpath(POLARSSLPATH, "include")]
        env["PINLIBS"] += ["polarssl"]
        env["CPPFLAGS"] += " -D_WITH_POLARSSL_=1 "

    env["CPPPATH"] += ["."]

    # HDF5
    env["PINLIBS"] += ["hdf5", "hdf5_hl"]

    # Harness needs these defined
    env["CPPFLAGS"] += ' -DPIN_PATH="' + joinpath(PINPATH, "intel64/bin/pinbin") + '" '
    env["CPPFLAGS"] += ' -DZSIM_PATH="' + joinpath(ROOT, joinpath(buildDir, "libzsim.so")) + '" '
    env["CPPFLAGS"] += ' -DLIBZSIM_NAME="libzsim.so" '

    # Do PGO?
    if pgo == "generate":
        genFlags = " -prof-gen " if useIcc else " -fprofile-generate "
        env["PINCPPFLAGS"] += genFlags
        env["PINLINKFLAGS"] += genFlags
    elif pgo == "use":
        if useIcc: useFlags = " -prof-use "
        else: useFlags = " -fprofile-use -fprofile-correction "
        # even single-threaded sims use internal threads, so we need correction
        env["PINCPPFLAGS"] += useFlags
        env["PINLINKFLAGS"] += useFlags

    env.SConscript("src/SConscript", variant_dir=buildDir, exports= {'env' : env.Clone()})

####

AddOption('--buildDir', dest='buildDir', type='string', default="build/", nargs=1, action='store', metavar='DIR', help='Base build directory')
AddOption('--d', dest='debugBuild', default=False, action='store_true', help='Do a debug build')
AddOption('--o', dest='optBuild', default=False, action='store_true', help='Do an opt build (optimized, with assertions and symbols)')
AddOption('--r', dest='releaseBuild', default=False, action='store_true', help='Do a release build (optimized, no assertions, no symbols)')
AddOption('--p', dest='pgoBuild', default=False, action='store_true', help='Enable PGO')
AddOption('--pgoPhase', dest='pgoPhase', default="none", action='store', help='PGO phase (just run with --p to do them all)')


baseBuildDir = GetOption('buildDir')
buildTypes = []
if GetOption('debugBuild'): buildTypes.append("debug")
if GetOption('releaseBuild'): buildTypes.append("release")
if GetOption('optBuild') or len(buildTypes) == 0: buildTypes.append("opt")

march = "core2" # ensure compatibility across condor nodes
#march = "native" # for profiling runs

buildFlags = {"debug": "-g -O0",
              "opt": "-march=%s -g -O3 -funroll-loops" % march, # unroll loops tends to help in zsim, but in general it can cause slowdown
              "release": "-march=%s -O3 -DNASSERT -funroll-loops -fweb" % march} # fweb saves ~4% exec time, but makes debugging a world of pain, so careful

pgoPhase = GetOption('pgoPhase')

# The PGO flow calls scons recursively. Hacky, but pretty much the only option:
# scons can't build the same file twice, and although gcc enables you to change
# the fprofile path, it considers the whole relative path as the filename
# (e.g., build/opt/zsim.os), and all hell breaks loose when it tries to create
# files in another dir. And because it uses checksums for filenames, it breaks
# when you move the files. Check the repo for a version that tries this.
if GetOption('pgoBuild'):
    for type in buildTypes:
        print("Building PGO binary")
        root = Dir('.').abspath
        testsDir = joinpath(root, "tests")
        trainCfgs = [f for f in os.listdir(testsDir) if f.startswith("pgo")]
        print("Using training configs", trainCfgs)

        baseDir = joinpath(baseBuildDir, "pgo-" + type)
        genCmd = "scons -j16 --pgoPhase=generate-" + type
        runCmds = []
        for cfg in trainCfgs:
            runCmd = "mkdir -p pgo-tmp && cd pgo-tmp && ../" + baseDir + "/zsim ../tests/" + cfg + " && cd .."
            runCmds.append(runCmd)
        useCmd = "scons -j16 --pgoPhase=use-" + type
        Environment(ENV = os.environ).Command("dummyTgt-" + type, [], " && ".join([genCmd] + runCmds + [useCmd]))
elif pgoPhase.startswith("generate"):
    type = pgoPhase.split("-")[1]
    buildSim(buildFlags[type], baseBuildDir, "pgo-" + type, "generate")
elif pgoPhase.startswith("use"):
    type = pgoPhase.split("-")[1]
    buildSim(buildFlags[type], baseBuildDir, "pgo-" + type, "use")
    baseDir = joinpath(baseBuildDir, "pgo-" + type)
    Depends(Glob(joinpath(baseDir, "*.os")), "pgo-tmp/zsim.out") #force a rebuild
else:
    for type in buildTypes:
        buildSim(buildFlags[type], baseBuildDir, type)
