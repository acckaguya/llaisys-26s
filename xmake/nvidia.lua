-- Generate native SASS and retain Ampere PTX for forward compatibility.
target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_values("cuda.rdc", false)
    add_cugencodes("native")
    add_cugencodes("compute_80")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_values("cuda.rdc", false)
    add_cugencodes("native")
    add_cugencodes("compute_80")
    add_links("cublas")

    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
