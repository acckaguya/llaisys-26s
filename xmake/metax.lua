-- xmake/metax.lua
-- MetaX (曦云 C500) 原生 MXMACA 设备支持。
-- 用 mxcc 编译 .mc 源文件，与 NVIDIA(CUDA) / CPU 目标完全隔离。
--
-- 实现方式：
--   1) target 的 on_load 钩子调用 mxcc 增量编译 .mc -> .o
--      （输出到 build/metax-objs/），然后 target:add("files", obj) 动态注册对象
--   2) add_rules("c++") 提供对象合并能力
--   3) 主 xmake.lua 给 llaisys 共享库目标链接 mcruntime/mcblas

local maca_path = os.getenv("MACA_PATH") or "/opt/maca"
local mxcc = path.join(maca_path, "mxgpu_llvm/bin/mxcc")
local metax_objdir = path.join(os.projectdir(), "build/metax-objs")

local maca_includes = {
    path.join(os.projectdir(), "include"),
    path.join(maca_path, "include/mcr"),
    path.join(maca_path, "include/mcblas"),
    path.join(maca_path, "include/common"),
    path.join(maca_path, "include"),
}

-- 链接配置（经 add_deps 传播到 llaisys 共享库，确保排在 metax 静态库之后，
-- 从而被 --as-needed 正确保留）
local maca_linkdirs = {
    path.join(maca_path, "lib"),
    path.join(maca_path, "lib64"),
}

target("llaisys-device-metax")
    set_kind("static")
    -- c++ 规则提供对象合并能力
    add_rules("c++")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_linkdirs(maca_linkdirs)
    add_links("mcruntime")

    on_load(function (target)
        os.mkdir(metax_objdir)
        for _, src in ipairs(os.files(path.join(os.projectdir(), "src/device/metax/*.mc"))) do
            local obj = path.join(metax_objdir, path.basename(src):gsub("%.mc$", "") .. ".o")
            if not os.isfile(obj) or os.mtime(obj) < os.mtime(src) then
                local argv = {
                    "-x", "maca", "-std=c++17", "-offload-arch", "native",
                    "-fPIC",
                    "--maca-path=" .. maca_path,
                }
                for _, inc in ipairs(maca_includes) do
                    table.insert(argv, "-I")
                    table.insert(argv, inc)
                end
                table.insert(argv, "-c")
                table.insert(argv, src)
                table.insert(argv, "-o")
                table.insert(argv, obj)
                os.execv(mxcc, argv)
            end
            target:add("files", obj)
        end
    end)

    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    -- c++ 规则提供对象合并能力
    add_rules("c++")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_linkdirs(maca_linkdirs)
    add_links("mcblas")

    on_load(function (target)
        os.mkdir(metax_objdir)
        for _, src in ipairs(os.files(path.join(os.projectdir(), "src/ops/*/metax/*.mc"))) do
            local obj = path.join(metax_objdir, path.basename(src):gsub("%.mc$", "") .. ".o")
            if not os.isfile(obj) or os.mtime(obj) < os.mtime(src) then
                local argv = {
                    "-x", "maca", "-std=c++17", "-offload-arch", "native",
                    "-fPIC",
                    "--maca-path=" .. maca_path,
                }
                for _, inc in ipairs(maca_includes) do
                    table.insert(argv, "-I")
                    table.insert(argv, inc)
                end
                table.insert(argv, "-c")
                table.insert(argv, src)
                table.insert(argv, "-o")
                table.insert(argv, obj)
                os.execv(mxcc, argv)
            end
            target:add("files", obj)
        end
    end)

    on_install(function (target) end)
target_end()
