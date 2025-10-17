/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#ifndef WebLocalizableStringsInternal_h
#define WebLocalizableStringsInternal_h

// This file should be used for localizing strings internal to the WebKit and WebCore frameworks.

#ifdef __cplusplus
extern "C" {
#endif

NSString *WebLocalizedStringInternal(const char* key) NS_FORMAT_ARGUMENT(1);

#ifdef __cplusplus
}
#endif

#define UI_STRING_INTERNAL(string, comment) WebLocalizedStringInternal(string)
#define UI_STRING_KEY_INTERNAL(string, key, comment) WebLocalizedStringInternal(key)

#endif // WebLocalizableStringsInternal_h
