def define_targets(rules, extra_defines=[]):
    rules.cc_library(
        name = "xpu",
        srcs = rules.glob(
            [
                "*.cpp",
                "impl/*.cpp",
            ],
            exclude = [
                "test/**/*.cpp",
            ],
        ),
        hdrs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
                "XPUMacros.h",
            ],
        ),
        defines = ["USE_XPU"] + extra_defines,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        target_compatible_with = rules.requires_xpu_enabled(),
        visibility = ["//visibility:public"],
        deps = [
            ":Macros",
            "//c10/core:base",
            "//c10/macros",
            "//c10/util:base",
            "@xpu",
        ],
        # This library uses registration. Don't let registered
        # entities be removed.
        alwayslink = True,
    )

    rules.cc_library(
        name = "Macros",
        srcs = [":xpu_cmake_macros"],
        hdrs = ["XPUMacros.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cmake_configure_file(
        name = "xpu_cmake_macros",
        src = "impl/xpu_cmake_macros.h.in",
        out = "impl/xpu_cmake_macros.h",
        definitions = [],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )
