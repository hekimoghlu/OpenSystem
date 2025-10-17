/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#ifndef WKAPICastPlayStation_h
#define WKAPICastPlayStation_h

#ifndef WKAPICast_h
#error "Please #include \"WKAPICast.h\" instead of this file directly."
#endif

namespace WebKit {

class PlayStationWebView;

WK_ADD_API_MAPPING(WKViewRef, PlayStationWebView)

template<>
inline WKViewRef toAPI<>(PlayStationWebView* view)
{
    return reinterpret_cast<WKViewRef>(static_cast<void*>(view));
}

template<>
inline PlayStationWebView* toImpl<>(WKViewRef view)
{
    return static_cast<PlayStationWebView*>(static_cast<void*>(const_cast<typename std::remove_const<typename std::remove_pointer<WKViewRef>::type>::type*>(view)));
}

}

#endif /* WKAPICastPlayStation_h */
