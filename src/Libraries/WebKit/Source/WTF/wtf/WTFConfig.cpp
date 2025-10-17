/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include <wtf/WTFConfig.h>

#include <wtf/Gigacage.h>
#include <wtf/Lock.h>
#include <wtf/MathExtras.h>
#include <wtf/PageBlock.h>
#include <wtf/StdLibExtras.h>

#if OS(DARWIN)
#include <dlfcn.h>
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#include <mach/vm_param.h>
#endif

#if PLATFORM(COCOA)
#include <wtf/spi/cocoa/MachVMSPI.h>
#include <mach/mach.h>
#elif OS(LINUX)
#include <sys/mman.h>
#endif

#if USE(APPLE_INTERNAL_SDK)
#include <WebKitAdditions/WTFConfigAdditions.h>
#endif

#include <mutex>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WebConfig {

alignas(WTF::ConfigAlignment) Slot g_config[WTF::ConfigSizeToProtect / sizeof(Slot)];

} // namespace WebConfig

#if !USE(SYSTEM_MALLOC)
static_assert(Gigacage::startSlotOfGigacageConfig == WebConfig::reservedSlotsForExecutableAllocator + WebConfig::additionalReservedSlots);
#endif

namespace WTF {

void setPermissionsOfConfigPage()
{
#if PLATFORM(COCOA)
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        mach_vm_address_t addr = std::bit_cast<uintptr_t>(static_cast<void*>(WebConfig::g_config));
        auto flags = VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE | VM_FLAGS_PERMANENT;

        auto attemptVMMapping = [&] {
            constexpr size_t preWTFConfigSize = Gigacage::startOffsetOfGigacageConfig + Gigacage::reservedBytesForGigacageConfig;

            // We may have potentially initialized some of g_config, namely the
            // gigacage config, prior to reaching this function. We need to
            // preserve these config contents across the mach_vm_map.
            uint8_t preWTFConfigContents[preWTFConfigSize];
            memcpySpan(std::span<uint8_t> { preWTFConfigContents, preWTFConfigSize }, std::span<uint8_t> { std::bit_cast<uint8_t*>(&WebConfig::g_config), preWTFConfigSize });
            auto result = mach_vm_map(mach_task_self(), &addr, ConfigSizeToProtect, pageSize() - 1, flags, MEMORY_OBJECT_NULL, 0, false, VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_DEFAULT);
            if (result == KERN_SUCCESS)
                memcpySpan(std::span<uint8_t> { std::bit_cast<uint8_t*>(&WebConfig::g_config), preWTFConfigSize }, std::span<uint8_t> { preWTFConfigContents, preWTFConfigSize });
            return result;
        };

        auto result = attemptVMMapping();

        if (result != KERN_SUCCESS) {
            flags &= ~VM_FLAGS_PERMANENT;
            result = attemptVMMapping();
        }

        RELEASE_ASSERT(result == KERN_SUCCESS);
    });
#endif // PLATFORM(COCOA)
}

void Config::initialize()
{
    // FIXME: We should do a placement new for Config so we can use default initializers.
    []() -> void {
        uintptr_t onePage = pageSize(); // At least, first one page must be unmapped.
#if OS(DARWIN)
#ifdef __LP64__
        using Header = struct mach_header_64;
#else
        using Header = struct mach_header;
#endif
        const auto* header = static_cast<const Header*>(dlsym(RTLD_MAIN_ONLY, MH_EXECUTE_SYM));
        if (header) {
            unsigned long size = 0;
            const auto* data = getsegmentdata(header, "__PAGEZERO", &size);
            if (!data && size) {
                // If __PAGEZERO starts with 0 address and it has size. [0, size] region cannot be
                // mapped for accessible pages.
                uintptr_t afterZeroPages = std::bit_cast<uintptr_t>(data) + size;
                g_wtfConfig.lowestAccessibleAddress = roundDownToMultipleOf(onePage, std::max<uintptr_t>(onePage, afterZeroPages));
                return;
            }
        }
#endif
        g_wtfConfig.lowestAccessibleAddress = onePage;
    }();
    g_wtfConfig.highestAccessibleAddress = static_cast<uintptr_t>((1ULL << OS_CONSTANT(EFFECTIVE_ADDRESS_WIDTH)) - 1);
    SignalHandlers::initialize();

    uint8_t* reservedConfigBytes = reinterpret_cast_ptr<uint8_t*>(WebConfig::g_config + WebConfig::reservedSlotsForExecutableAllocator);

#if USE(APPLE_INTERNAL_SDK)
    WTF_INITIALIZE_ADDITIONAL_CONFIG();
#endif

    const char* useAllocationProfilingRaw = getenv("JSC_useAllocationProfiling");
    if (useAllocationProfilingRaw) {
        auto useAllocationProfiling = unsafeSpan(useAllocationProfilingRaw);
        if (equalLettersIgnoringASCIICase(useAllocationProfiling, "true"_s)
            || equalLettersIgnoringASCIICase(useAllocationProfiling, "yes"_s)
            || equal(useAllocationProfiling, "1"_s))
            reservedConfigBytes[WebConfig::ReservedByteForAllocationProfiling] = 1;
        else if (equalLettersIgnoringASCIICase(useAllocationProfiling, "false"_s)
            || equalLettersIgnoringASCIICase(useAllocationProfiling, "no"_s)
            || equal(useAllocationProfiling, "0"_s))
            reservedConfigBytes[WebConfig::ReservedByteForAllocationProfiling] = 0;

        const char* useAllocationProfilingModeRaw = getenv("JSC_allocationProfilingMode");
        if (useAllocationProfilingModeRaw && reservedConfigBytes[WebConfig::ReservedByteForAllocationProfiling] == 1) {
            unsigned value { 0 };
            if (sscanf(useAllocationProfilingModeRaw, "%u", &value) == 1) {
                RELEASE_ASSERT(value <= 0xFF);
                reservedConfigBytes[WebConfig::ReservedByteForAllocationProfilingMode] = static_cast<uint8_t>(value & 0xFF);
            }
        }
    }

}

void Config::finalize()
{
    static std::once_flag once;
    std::call_once(once, [] {
        SignalHandlers::finalize();
        if (!g_wtfConfig.disabledFreezingForTesting)
            Config::permanentlyFreeze();
    });
}

void Config::permanentlyFreeze()
{
    RELEASE_ASSERT(roundUpToMultipleOf(pageSize(), ConfigSizeToProtect) == ConfigSizeToProtect);
    ASSERT(!g_wtfConfig.disabledFreezingForTesting);

    if (!g_wtfConfig.isPermanentlyFrozen) {
        g_wtfConfig.isPermanentlyFrozen = true;
#if GIGACAGE_ENABLED
        g_gigacageConfig.isPermanentlyFrozen = true;
#endif
    }

    int result = 0;

#if PLATFORM(COCOA)
    enum {
        DontUpdateMaximumPermission = false,
        UpdateMaximumPermission = true
    };

    // There's no going back now!
    result = vm_protect(mach_task_self(), reinterpret_cast<vm_address_t>(&WebConfig::g_config), ConfigSizeToProtect, UpdateMaximumPermission, VM_PROT_READ);
#elif OS(LINUX)
    result = mprotect(&WebConfig::g_config, ConfigSizeToProtect, PROT_READ);
#elif OS(WINDOWS)
    // FIXME: Implement equivalent for Windows, maybe with VirtualProtect.
    // Also need to fix WebKitTestRunner.
#endif

    RELEASE_ASSERT(!result);
    RELEASE_ASSERT(g_wtfConfig.isPermanentlyFrozen);
}

void Config::disableFreezingForTesting()
{
    RELEASE_ASSERT(!g_wtfConfig.isPermanentlyFrozen);
    g_wtfConfig.disabledFreezingForTesting = true;
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
