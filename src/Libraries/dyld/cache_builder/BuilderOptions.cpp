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
#include "BuilderOptions.h"
#include "Platform.h"

using namespace cache_builder;
using dyld3::GradedArchs;

using mach_o::Platform;

//
// MARK: --- cache_builder::Options methods ---
//

BuilderOptions::BuilderOptions(std::string_view archName, Platform platform,
                               bool dylibsRemovedFromDisk, bool isLocallyBuiltCache,
                               CacheKind kind, bool forceDevelopmentSubCacheSuffix)
    : archs(GradedArchs::forName(archName.data()))
    , platform(platform)
    , dylibsRemovedFromDisk(dylibsRemovedFromDisk)
    , isLocallyBuiltCache(isLocallyBuiltCache)
    , forceDevelopmentSubCacheSuffix(forceDevelopmentSubCacheSuffix)
    , kind(kind)
{
}

bool BuilderOptions::isSimulator() const
{
    return this->platform.isSimulator();
}

bool BuilderOptions::isExclaveKit() const
{
    return this->platform.isExclaveKit();
}

//
// MARK: --- cache_builder::InputFile methods ---
//

void InputFile::addError(error::Error&& err)
{
    // This is a good place to breakpoint and catch if a specific dylib has an error
    this->errors.push_back(std::move(err));
}

std::span<const error::Error> InputFile::getErrors() const
{
    return this->errors;
}

bool InputFile::hasError() const
{
    return !this->errors.empty();
}
