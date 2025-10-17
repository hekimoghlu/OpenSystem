/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#pragma once

#include <JavaScriptCore/SourceProvider.h>

#include <wtf/HashMap.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class RunAsAsyncFunction : bool { No, Yes };
enum class ForceUserGesture : bool { No, Yes };
enum class RemoveTransientActivation : bool { No, Yes };

using ArgumentWireBytesMap = HashMap<String, Vector<uint8_t>>;

struct RunJavaScriptParameters {
    RunJavaScriptParameters(String&& source, JSC::SourceTaintedOrigin taintedness, URL&& sourceURL, RunAsAsyncFunction runAsAsyncFunction, std::optional<ArgumentWireBytesMap>&& arguments, ForceUserGesture forceUserGesture, RemoveTransientActivation removeTransientActivation)
        : source(WTFMove(source))
        , taintedness(taintedness)
        , sourceURL(WTFMove(sourceURL))
        , runAsAsyncFunction(runAsAsyncFunction)
        , arguments(WTFMove(arguments))
        , forceUserGesture(forceUserGesture)
        , removeTransientActivation(removeTransientActivation)
    {
    }

    RunJavaScriptParameters(const String& source, JSC::SourceTaintedOrigin taintedness, URL&& sourceURL, bool runAsAsyncFunction, std::optional<ArgumentWireBytesMap>&& arguments, bool forceUserGesture, RemoveTransientActivation removeTransientActivation)
        : source(source)
        , taintedness(taintedness)
        , sourceURL(WTFMove(sourceURL))
        , runAsAsyncFunction(runAsAsyncFunction ? RunAsAsyncFunction::Yes : RunAsAsyncFunction::No)
        , arguments(WTFMove(arguments))
        , forceUserGesture(forceUserGesture ? ForceUserGesture::Yes : ForceUserGesture::No)
        , removeTransientActivation(removeTransientActivation)
    {
    }

    RunJavaScriptParameters(String&& source, JSC::SourceTaintedOrigin taintedness, URL&& sourceURL, bool runAsAsyncFunction, std::optional<ArgumentWireBytesMap>&& arguments, bool forceUserGesture, RemoveTransientActivation removeTransientActivation)
        : source(WTFMove(source))
        , taintedness(taintedness)
        , sourceURL(WTFMove(sourceURL))
        , runAsAsyncFunction(runAsAsyncFunction ? RunAsAsyncFunction::Yes : RunAsAsyncFunction::No)
        , arguments(WTFMove(arguments))
        , forceUserGesture(forceUserGesture ? ForceUserGesture::Yes : ForceUserGesture::No)
        , removeTransientActivation(removeTransientActivation)
    {
    }

    String source;
    JSC::SourceTaintedOrigin taintedness;
    URL sourceURL;
    RunAsAsyncFunction runAsAsyncFunction;
    std::optional<ArgumentWireBytesMap> arguments;
    ForceUserGesture forceUserGesture;
    RemoveTransientActivation removeTransientActivation;
};

} // namespace WebCore
