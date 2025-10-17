/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#ifndef mach_o_FunctionVariantsWriter_h
#define mach_o_FunctionVariantsWriter_h

// posix
#include <stdarg.h>

// stl
#include <span>

// mach_o
#include "Defines.h"
#include "Error.h"
#include "FunctionVariants.h"


namespace mach_o {


/*!
 * @struct FunctionVariantsRuntimeTableWriter
 *
 * @abstract
 *      Table for all variants of one function
 */
struct VIS_HIDDEN FunctionVariantsRuntimeTableWriter : public FunctionVariantsRuntimeTable
{
    static FunctionVariantsRuntimeTableWriter*  make(FunctionVariantsRuntimeTable::Kind kind, size_t variantsCount);
    Error                                       setEntry(size_t index, uint32_t implOffset, bool implIsTable, std::span<const uint8_t> flagIndexes);
};


/*!
 * @struct FunctionVariantsWriter
 *
 * @abstract
 *      Wrapper for building a FunctionVariantsRuntimeTable in an image.
 *      Located in LINKEDIT.
 *      Pointed to by `LC_FUNCTION_VARIANTS`
 *
 */
struct VIS_HIDDEN FunctionVariantsWriter : public FunctionVariants
{
public:
                                FunctionVariantsWriter(std::span<const FunctionVariantsRuntimeTable*> entries);
    std::span<const uint8_t>    bytes() const { return _builtBytes; }

private:
    std::vector<uint8_t> _builtBytes;
    Error                _buildError;
};



/*!
 * @struct FunctionVariantFixupsWriter
 *
 * @abstract
 *      Wrapper for building uses of non-exported function variants.
 *      Located in LINKEDIT.
 *      Pointed to by `LC_FUNCTION_VARIANT_FIXUPS`
 *
 */
struct VIS_HIDDEN FunctionVariantFixupsWriter : public FunctionVariantFixups
{
public:
                                // used to build linkedit content
                                FunctionVariantFixupsWriter(std::span<const InternalFixup> entries);
    std::span<const uint8_t>    bytes() const { return _builtBytes; }

private:
    std::vector<uint8_t> _builtBytes;
};



} // namespace mach_o

#endif /* mach_o_FunctionVariantsWriter_h */
