/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef mach_o_FunctionVariants_h
#define mach_o_FunctionVariants_h

#include <stdarg.h>

#include <span>

#include "Defines.h"
#include "Error.h"

namespace mach_o {


/*!
 * @struct FunctionVariantsRuntimeTable
 *
 * @abstract
 *      Table for all variants of one function
 */
struct VIS_HIDDEN FunctionVariantsRuntimeTable
{
    enum class Kind : uint32_t { perProcess=1, systemWide=2, arm64=3, x86_64=4 };
    uint32_t    forEachVariant(void (^callback)(Kind kind, uint32_t implOffset, bool implIsTable, std::span<const uint8_t> flagIndexes, bool& stop)) const;
    uint32_t    size() const;
    Error       valid(size_t length) const;

    Kind        kind;
    uint32_t    count;
    struct {
        uint32_t    impl         : 31,  // offset to function or index of another table
                    anotherTable :  1;  // impl is index of another FunctionVariantsRuntimeTable
        uint8_t     flagBitNums[4];
    }           entries[1];
};


/*!
 * @struct FunctionVariants
 *
 * @abstract
 *      Wrapper for all FunctionVariantsRuntimeTable in the image.
 *      Located in LINKEDIT.
 *      Pointed to by `LC_FUNCTION_VARIANTS`
 *
 */
struct VIS_HIDDEN FunctionVariants
{
public:
                                        // construct from a chunk of LINKEDIT
                                        FunctionVariants(std::span<const uint8_t> linkeditBytes);

    Error                               valid() const;
    uint32_t                            count() const;
    const FunctionVariantsRuntimeTable* entry(uint32_t index) const;

protected:
    // only for use by NListSymbolTableWriter
    FunctionVariants() = default;

    struct OnDiskFormat
    {
        uint32_t  tableCount;       // number of FunctionVariantsRuntimeTable in this Linkedit blob
        uint32_t  tableOffsets[];   // offset to start of each FunctionVariantsRuntimeTable within this blob
        // FunctionVariantsRuntimeTable
    };

    OnDiskFormat*   header() const;

    std::span<const uint8_t> _bytes;
};



/*!
 * @struct FunctionVariantFixups
 *
 * @abstract
 *      Wrapper for any uses of non-exported function variants.
 *      Located in LINKEDIT.
 *      Pointed to by `LC_FUNCTION_VARIANT_FIXUPS`
 *
 * @discussion
 *      If there is a call to a variant function within the same linkage unit, the linker will generate
 *      a "stub" which jumps through a GOT. That GOT slot needs to be set to the correct variant at
 *      load time by dyld.  This is done in two ways:
 *      1) If the varianted function is exported, the linker will set the GOT slot to be a bind-to-self
 *      of the symbol name. The lookup of that name will resolve which variant to use.
 *      2) If the varianted function is not exported (purely internal function), the linker will set the
 *      GOT to be a rebase to the "default" variant, and add an InternalFixup to `LC_FUNCTION_VARIANT_FIXUPS`.
 *      This allows the binary to run on older OSs and just use the default variant.  When run on newer OS
 *      versions that understand `LC_FUNCTION_VARIANT_FIXUPS`, dyld (after the rebase is done), will process
 *      each InternalFixup, and overwrite the GOT slot with the best variant.
 *
 */
struct VIS_HIDDEN FunctionVariantFixups
{
public:
                                        // construct from a chunk of LINKEDIT
                                        FunctionVariantFixups(std::span<const uint8_t> linkeditBytes);

    struct InternalFixup
    {
        uint32_t segOffset;
        uint32_t segIndex     :  4,     // segIndex and segOffset are location of GOT slot to update
                 variantIndex :  8,     // index into FunctionVariants (target of this fixup)
                 pacAuth      :  1,     // PAC signed or not
                 pacAddress   :  1,
                 pacKey       :  2,
                 pacDiversity : 16;

        bool     operator==(const InternalFixup&) const = default;
    };
    static_assert(sizeof(InternalFixup) == 8, "bit field wrong size");

    void                                forEachFixup(void (^callback)(InternalFixup fixupInfo)) const;
protected:
    FunctionVariantFixups() = default; // for use by FunctionVariantFixupsWriter

    std::span<const InternalFixup> _fixups;
};



} // namespace mach_o

#endif /* mach_o_FunctionVariants_h */
