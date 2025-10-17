/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

#include "SpeechRecognitionResult.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SpeechRecognitionResultList final : public RefCounted<SpeechRecognitionResultList> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechRecognitionResultList);
public:
    static Ref<SpeechRecognitionResultList> create();
    static Ref<SpeechRecognitionResultList> create(Vector<Ref<SpeechRecognitionResult>>&&);

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    size_t length() const { return m_list.size(); }
    SpeechRecognitionResult* item(uint64_t index) const;

    void add(SpeechRecognitionResult&);

private:
    SpeechRecognitionResultList() = default;
    explicit SpeechRecognitionResultList(Vector<Ref<SpeechRecognitionResult>>&&);

    Vector<Ref<SpeechRecognitionResult>> m_list;
};

} // namespace WebCore
