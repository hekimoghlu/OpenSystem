/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#ifndef MachOLoaded_h
#define MachOLoaded_h

#include <stdint.h>

#include "Array.h"
#include "MachOFile.h"

namespace dyld3 {


// A mach-o mapped into memory with zero-fill expansion
// Can be used in dyld at runtime or during closure building
struct VIS_HIDDEN MachOLoaded : public MachOFile
{
	typedef const MachOLoaded* (^DependentToMachOLoaded)(const MachOLoaded* image, uint32_t depIndex);

    // for dlsym()
	bool                hasExportedSymbol(const char* symbolName, DependentToMachOLoaded finder, void** result,
                                          bool* resultPointsToInstructions) const;

    // for DYLD_PRINT_SEGMENTS
    std::string_view    segmentName(uint32_t segIndex) const;

    // used to see if main executable overlaps shared region
    bool                intersectsRange(uintptr_t start, uintptr_t length) const;

    // for _dyld_get_image_slide()
    intptr_t            getSlide() const;

    // for dladdr()
    bool                findClosestSymbol(uint64_t unSlidAddr, const char** symbolName, uint64_t* symbolUnslidAddr) const;

    // for _dyld_find_unwind_sections()
    const void*         findSectionContent(const char* segName, const char* sectName, uint64_t& size) const;

    // used by dyld/libdyld to apply fixups
//#if BUILDING_DYLD || BUILDING_LIBDYLD
    void                    fixupAllChainedFixups(Diagnostics& diag, const dyld_chained_starts_in_image* starts, uintptr_t slide,
                                                  Array<const void*> bindTargets, void (^fixupLogger)(void* loc, void* newValue)) const;
//#endif

    void                    forEachGlobalSymbol(Diagnostics& diag, void (^callback)(const char* symbolName, uint64_t n_value, uint8_t n_type, uint8_t n_sect, uint16_t n_desc, bool& stop)) const;

    void                    forEachImportedSymbol(Diagnostics& diag, void (^callback)(const char* symbolName, uint64_t n_value, uint8_t n_type, uint8_t n_sect, uint16_t n_desc, bool& stop)) const;

    struct LayoutInfo {
        uintptr_t    slide;
        uintptr_t    textUnslidVMAddr;
        uintptr_t    linkeditUnslidVMAddr;
        uint32_t     linkeditFileOffset;
        uint32_t     linkeditFileSize;
        uint32_t     linkeditSegIndex;
        uint32_t     lastSegIndex;
    };

    struct LinkEditInfo
    {
        const dyld_info_command*        dyldInfo;
        const linkedit_data_command*    exportsTrie;
        const linkedit_data_command*    chainedFixups;
        const symtab_command*           symTab;
        const dysymtab_command*         dynSymTab;
        const linkedit_data_command*    splitSegInfo;
        const linkedit_data_command*    functionStarts;
        const linkedit_data_command*    dataInCode;
        const linkedit_data_command*    codeSig;
        LayoutInfo                      layout;
    };
    void                    getLinkEditPointers(Diagnostics& diag, LinkEditInfo&) const;
    uint64_t                firstSegmentFileOffset() const;

    void forEachFixupInSegmentChains(Diagnostics& diag, const dyld_chained_starts_in_segment* segInfo, bool notifyNonPointers,
                                     void (^handler)(ChainedFixupPointerOnDisk* fixupLocation, const dyld_chained_starts_in_segment* segInfo, bool& stop)) const;

    // for dyld loaded images
    void                    forEachFixupInAllChains(Diagnostics& diag, const dyld_chained_starts_in_image* starts, bool notifyNonPointers,
                                                    void (^callback)(ChainedFixupPointerOnDisk* fixupLocation, const dyld_chained_starts_in_segment* segInfo, bool& stop)) const;
    // for preload images
    void                    forEachFixupInAllChains(Diagnostics& diag, uint16_t pointer_format, uint32_t starts_count, const uint32_t chain_starts[],
                                                    void (^handler)(ChainedFixupPointerOnDisk* fixupLocation, bool& stop)) const;

    void                    forEachFunctionStart(void (^callback)(uint64_t runtimeOffset)) const;

protected:

    struct FoundSymbol {
        enum class Kind { headerOffset, absolute, resolverOffset };
        Kind                kind;
        bool                isThreadLocal;
        bool                isWeakDef;
        const MachOLoaded*  foundInDylib;
        uint64_t            value;
        uint32_t            resolverFuncOffset;
        const char*         foundSymbolName;
    };

    bool                    findExportedSymbol(Diagnostics& diag, const char* symbolName, bool weakImport, FoundSymbol& foundInfo, DependentToMachOLoaded finder) const;

    void                    getLinkEditLoadCommands(Diagnostics& diag, LinkEditInfo& result) const;
    void                    getLayoutInfo(LayoutInfo&) const;
    const uint8_t*          getLinkEditContent(const LayoutInfo& info, uint32_t fileOffset) const;
    const uint8_t*          getExportsTrie(const LinkEditInfo& info, uint64_t& trieSize) const;
    void                    forEachLocalSymbol(Diagnostics& diag, void (^callback)(const char* symbolName, uint64_t n_value, uint8_t n_type, uint8_t n_sect, uint16_t n_desc, bool& stop)) const;

    bool                    findClosestFunctionStart(uint64_t address, uint64_t* functionStartAddress) const;

};

} // namespace dyld3

#endif /* MachOLoaded_h */
