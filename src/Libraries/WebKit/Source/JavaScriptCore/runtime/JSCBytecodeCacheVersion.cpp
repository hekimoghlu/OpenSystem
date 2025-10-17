/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include "JSCBytecodeCacheVersion.h"

#include <wtf/DataLog.h>
#include <wtf/HexNumber.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/SuperFastHash.h>

#if OS(UNIX)
#include <dlfcn.h>
#if OS(DARWIN)
#include <mach-o/dyld.h>
#include <uuid/uuid.h>
#include <wtf/spi/darwin/dyldSPI.h>
#else
#include <link.h>
#endif
#endif

namespace JSC {

namespace JSCBytecodeCacheVersionInternal {
static constexpr bool verbose = false;
}

uint32_t computeJSCBytecodeCacheVersion()
{
    UNUSED_VARIABLE(JSCBytecodeCacheVersionInternal::verbose);
    static LazyNeverDestroyed<uint32_t> cacheVersion;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        void* jsFunctionAddr = std::bit_cast<void*>(&computeJSCBytecodeCacheVersion);
#if OS(DARWIN)
        uuid_t uuid;
        if (const mach_header* header = dyld_image_header_containing_address(jsFunctionAddr); header && _dyld_get_image_uuid(header, uuid)) {
            uuid_string_t uuidString = { };
            uuid_unparse(uuid, uuidString);
            cacheVersion.construct(SuperFastHash::computeHash(uuidString));
            dataLogLnIf(JSCBytecodeCacheVersionInternal::verbose, "UUID of JavaScriptCore.framework:", uuidString);
            return;
        }
        cacheVersion.construct(0);
        dataLogLnIf(JSCBytecodeCacheVersionInternal::verbose, "Failed to get UUID for JavaScriptCore framework");
#elif OS(UNIX) && !PLATFORM(PLAYSTATION)
        auto result = ([&] -> std::optional<uint32_t> {
            Dl_info info { };
            if (!dladdr(jsFunctionAddr, &info))
                return std::nullopt;

            if (!info.dli_fbase)
                return std::nullopt;

            struct DLParam {
                void* start { nullptr };
                std::span<const uint8_t> description;
            };

            DLParam param { };
            param.start = info.dli_fbase;
            if (!dl_iterate_phdr(static_cast<int(*)(struct dl_phdr_info*, size_t, void*)>(
                [](struct dl_phdr_info* info, size_t, void* priv) -> int {
                    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Unix port
                    auto* data = static_cast<DLParam*>(priv);
                    void* start = nullptr;
                    for (unsigned i = 0; i < info->dlpi_phnum; ++i) {
                        if (info->dlpi_phdr[i].p_type == PT_LOAD) {
                            start = std::bit_cast<void*>(static_cast<uintptr_t>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr));
                            break;
                        }
                    }

                    if (start != data->start)
                        return 0;

                    for (unsigned i = 0; i < info->dlpi_phnum; ++i) {
                        if (info->dlpi_phdr[i].p_type != PT_NOTE)
                            continue;

                        // https://refspecs.linuxbase.org/elf/gabi4+/ch5.pheader.html#note_section
                        using NoteHeader = ElfW(Nhdr);

                        auto* payload = std::bit_cast<uint8_t*>(static_cast<uintptr_t>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr));
                        size_t length = info->dlpi_phdr[i].p_filesz;
                        for (size_t index = 0; index < length;) {
                            auto* cursor  = payload + index;
                            if ((index + sizeof(NoteHeader)) > length)
                                return 0;

                            auto* note = std::bit_cast<NoteHeader*>(cursor);
                            size_t size = sizeof(NoteHeader) + roundUpToMultipleOf<4>(note->n_namesz) + roundUpToMultipleOf<4>(note->n_descsz);
                            if ((index + size) > length)
                                return 0;

                            auto* name = cursor + sizeof(NoteHeader);
                            auto* description = cursor + sizeof(NoteHeader) + roundUpToMultipleOf<4>(note->n_namesz);

                            if (note->n_type == NT_GNU_BUILD_ID && note->n_descsz != 0 && note->n_namesz == 4 && memcmp(name, "GNU", 4) == 0) {
                                // Found build-id note.
                                data->description = std::span { description, note->n_descsz };
                                return 1;
                            }

                            index += size;
                        }
                    }
                    return 0;
                    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
                }), &param))
                    return std::nullopt;

                if (param.description.empty())
                    return std::nullopt;

                if constexpr (JSCBytecodeCacheVersionInternal::verbose) {
                    for (uint8_t value : param.description)
                        dataLog(hex(value));
                    dataLogLn("");
                }

                return SuperFastHash::computeHash(param.description);
        }());
        if (result) {
            cacheVersion.construct(result.value());
            return;
        }
        cacheVersion.construct(0);
        dataLogLnIf(JSCBytecodeCacheVersionInternal::verbose, "Failed to get UUID for JavaScriptCore framework");
#else
        UNUSED_VARIABLE(jsFunctionAddr);
        static constexpr uint32_t precomputedCacheVersion = SuperFastHash::computeHash(__TIMESTAMP__);
        cacheVersion.construct(precomputedCacheVersion);
#endif
    });
    return cacheVersion.get();
}

} // namespace JSC
