/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#include "FrameIdentifier.h"
#include "SpeechRecognitionConnectionClientIdentifier.h"

namespace WebCore {

class SpeechRecognitionConnectionClient;
class SpeechRecognitionUpdate;
struct ClientOrigin;

class SpeechRecognitionConnection : public RefCounted<SpeechRecognitionConnection> {
public:
    virtual ~SpeechRecognitionConnection() { }
    virtual void registerClient(SpeechRecognitionConnectionClient&) = 0;
    virtual void unregisterClient(SpeechRecognitionConnectionClient&) = 0;
    virtual void start(SpeechRecognitionConnectionClientIdentifier, const String& lang, bool continuous, bool interimResults, uint64_t maxAlternatives, ClientOrigin&&, FrameIdentifier) = 0;
    virtual void stop(SpeechRecognitionConnectionClientIdentifier) = 0;
    virtual void abort(SpeechRecognitionConnectionClientIdentifier) = 0;
    virtual void didReceiveUpdate(SpeechRecognitionUpdate&&) = 0;
};

} // namespace WebCore

