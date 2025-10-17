/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#include "CustomEffect.h"

#include "CustomEffectCallback.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/Exception.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CustomEffect);

ExceptionOr<Ref<CustomEffect>> CustomEffect::create(Document& document, Ref<CustomEffectCallback>&& callback, std::optional<std::variant<double, EffectTiming>>&& options)
{
    auto customEffect = adoptRef(*new CustomEffect(WTFMove(callback)));

    if (options) {
        OptionalEffectTiming timing;
        auto optionsValue = options.value();
        if (std::holds_alternative<double>(optionsValue)) {
            std::variant<double, String> duration = std::get<double>(optionsValue);
            timing.duration = duration;
        } else {
            auto effectTimingOptions = std::get<EffectTiming>(optionsValue);

            auto convertedDuration = effectTimingOptions.durationAsDoubleOrString();
            if (!convertedDuration)
                return Exception { ExceptionCode::TypeError };

            timing = {
                *convertedDuration,
                effectTimingOptions.iterations,
                effectTimingOptions.delay,
                effectTimingOptions.endDelay,
                effectTimingOptions.iterationStart,
                effectTimingOptions.easing,
                effectTimingOptions.fill,
                effectTimingOptions.direction
            };
        }
        auto updateTimingResult = customEffect->updateTiming(document, timing);
        if (updateTimingResult.hasException())
            return updateTimingResult.releaseException();
    }

    return customEffect;
}

CustomEffect::CustomEffect(Ref<CustomEffectCallback>&& callback)
    : m_callback(WTFMove(callback))
{
}

void CustomEffect::animationDidTick()
{
    auto computedTiming = getComputedTiming();
    if (!computedTiming.progress)
        return;

    m_callback->handleEvent(*computedTiming.progress);
}

} // namespace WebCore
