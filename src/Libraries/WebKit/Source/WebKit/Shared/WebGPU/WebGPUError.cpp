/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#include "WebGPUError.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUError.h>

namespace WebKit::WebGPU {

std::optional<Error> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Error& error)
{
    return WTF::switchOn(error, [this] (const Ref<WebCore::WebGPU::OutOfMemoryError>& outOfMemoryError) -> std::optional<Error> {
        auto result = convertToBacking(outOfMemoryError.get());
        if (!result)
            return std::nullopt;
        return { { *result } };
    }, [this] (const Ref<WebCore::WebGPU::ValidationError>& validationError) -> std::optional<Error> {
        auto result = convertToBacking(validationError.get());
        if (!result)
            return std::nullopt;
        return { { *result } };
    }, [this] (const Ref<WebCore::WebGPU::InternalError>& internalError) -> std::optional<Error> {
        auto result = convertToBacking(internalError.get());
        if (!result)
            return std::nullopt;
        return { { *result } };
    });
}

std::optional<WebCore::WebGPU::Error> ConvertFromBackingContext::convertFromBacking(const Error& error)
{
    return WTF::switchOn(error, [this] (const OutOfMemoryError& outOfMemoryError) -> std::optional<WebCore::WebGPU::Error> {
        auto result = convertFromBacking(outOfMemoryError);
        if (!result)
            return std::nullopt;
        return { result.releaseNonNull() };
    }, [this] (const ValidationError& validationError) -> std::optional<WebCore::WebGPU::Error> {
        auto result = convertFromBacking(validationError);
        if (!result)
            return std::nullopt;
        return { result.releaseNonNull() };
    }, [this] (const InternalError& internalError) -> std::optional<WebCore::WebGPU::Error> {
        auto result = convertFromBacking(internalError);
        if (!result)
            return std::nullopt;
        return { result.releaseNonNull() };
    });
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
