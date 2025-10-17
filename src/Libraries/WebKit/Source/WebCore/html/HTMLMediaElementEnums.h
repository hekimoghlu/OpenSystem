/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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

#include "MediaPlayerEnums.h"

namespace WebCore {

class HTMLMediaElementEnums : public MediaPlayerEnums {
public:
    using MediaPlayerEnums::VideoFullscreenMode;

    enum ReadyState { HAVE_NOTHING, HAVE_METADATA, HAVE_CURRENT_DATA, HAVE_FUTURE_DATA, HAVE_ENOUGH_DATA };
    enum NetworkState { NETWORK_EMPTY, NETWORK_IDLE, NETWORK_LOADING, NETWORK_NO_SOURCE };
    enum TextTrackVisibilityCheckType { CheckTextTrackVisibility, AssumeTextTrackVisibilityChanged };
    enum class InvalidURLAction : bool { DoNothing, Complain };

    typedef enum {
        NoSeek,
        Fast,
        Precise
    } SeekType;
};

String convertEnumerationToString(HTMLMediaElementEnums::ReadyState);
String convertEnumerationToString(HTMLMediaElementEnums::NetworkState);
String convertEnumerationToString(HTMLMediaElementEnums::TextTrackVisibilityCheckType);

} // namespace WebCore

namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebCore::HTMLMediaElementEnums::ReadyState> {
    static String toString(const WebCore::HTMLMediaElementEnums::ReadyState state)
    {
        return convertEnumerationToString(state);
    }
};

template <>
struct LogArgument<WebCore::HTMLMediaElementEnums::NetworkState> {
    static String toString(const WebCore::HTMLMediaElementEnums::NetworkState state)
    {
        return convertEnumerationToString(state);
    }
};

template <>
struct LogArgument<WebCore::HTMLMediaElementEnums::TextTrackVisibilityCheckType> {
    static String toString(const WebCore::HTMLMediaElementEnums::TextTrackVisibilityCheckType type)
    {
        return convertEnumerationToString(type);
    }
};

}; // namespace WTF

