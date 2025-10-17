/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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

#include <string_view>

static std::string_view staticStringView{"abc210"};
static std::string_view staticEmptyStringView{""};
static std::string_view staticNonASCIIStringView{"Ñ‚ÐµÑÑ‚"};

// UTF-16
static std::u16string_view staticU16StringView{u"abc210"};
static std::u16string_view staticU16EmptyStringView{u""};
static std::u16string_view staticU16NonASCIIStringView{u"Ñ‚ÐµÑÑ‚"};

// UTF-32
static std::u32string_view staticU32StringView{U"abc210"};
static std::u32string_view staticU32EmptyStringView{U""};
static std::u32string_view staticU32NonASCIIStringView{U"Ñ‚ÐµÑÑ‚"};
