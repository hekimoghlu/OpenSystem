/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#import <wtf/Assertions.h>

#ifndef LOG_CHANNEL_PREFIX
#define LOG_CHANNEL_PREFIX WebKitLog
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

#define WEBKIT_LOG_CHANNELS(M) \
    M(BackForward) \
    M(Bindings) \
    M(CacheSizes) \
    M(DocumentLoad) \
    M(Download) \
    M(Encoding) \
    M(Events) \
    M(FileDatabaseActivity) \
    M(FontCache) \
    M(FontSelection) \
    M(FontSubstitution) \
    M(FormDelegate) \
    M(History) \
    M(IconDatabase) \
    M(Loading) \
    M(PageCache) \
    M(PluginEvents) \
    M(Plugins) \
    M(Progress) \
    M(Redirect) \
    M(RemoteInspector) \
    M(TextInput) \
    M(Timing) \
    M(View) \

WEBKIT_LOG_CHANNELS(DECLARE_LOG_CHANNEL)

#undef DECLARE_LOG_CHANNEL

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

#ifdef __cplusplus
}
#endif
