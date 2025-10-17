/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#import "config.h"
#import "WKVisibilityPropagationView.h"

#if HAVE(VISIBILITY_PROPAGATION_VIEW) && USE(EXTENSIONKIT)

#import "AuxiliaryProcessProxy.h"
#import "ExtensionKitSPI.h"
#import "Logging.h"
#import <wtf/RetainPtr.h>
#import <wtf/WeakPtr.h>

namespace WebKit {
using ProcessAndInteractionPair = std::pair<WeakPtr<AuxiliaryProcessProxy>, RetainPtr<id<UIInteraction>>>;
}

@implementation WKVisibilityPropagationView {
    Vector<WebKit::ProcessAndInteractionPair> _processesAndInteractions;
}

- (void)propagateVisibilityToProcess:(WebKit::AuxiliaryProcessProxy&)process
{
    if ([self _containsInteractionForProcess:process])
        return;

    auto extensionProcess = process.extensionProcess();
    if (!extensionProcess)
        return;

    auto visibilityPropagationInteraction = extensionProcess->createVisibilityPropagationInteraction();
    if (!visibilityPropagationInteraction)
        return;

    [self addInteraction:visibilityPropagationInteraction.get()];

    RELEASE_LOG(Process, "Created visibility propagation interaction %@ for process with PID=%d", visibilityPropagationInteraction.get(), process.processID());

    auto processAndInteraction = std::make_pair(WeakPtr(process), visibilityPropagationInteraction);
    _processesAndInteractions.append(WTFMove(processAndInteraction));
}

- (void)stopPropagatingVisibilityToProcess:(WebKit::AuxiliaryProcessProxy&)process
{
    _processesAndInteractions.removeAllMatching([&](auto& processAndInteraction) {
        auto existingProcess = processAndInteraction.first.get();
        if (existingProcess && existingProcess != &process)
            return false;

        RELEASE_LOG(Process, "Removing visibility propagation interaction %p", processAndInteraction.second.get());

        [self removeInteraction:processAndInteraction.second.get()];
        return true;
    });
}

- (BOOL)_containsInteractionForProcess:(WebKit::AuxiliaryProcessProxy&)process
{
    return _processesAndInteractions.containsIf([&](auto& processAndInteraction) {
        return processAndInteraction.first.get() == &process;
    });
}

@end

#endif // HAVE(VISIBILITY_PROPAGATION_VIEW) && USE(EXTENSIONKIT)
