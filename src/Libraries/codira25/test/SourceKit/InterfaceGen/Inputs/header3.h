/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#define CF_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define CF_OPTIONS(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_ENUM(_type, _name) CF_ENUM(_type, _name)
#define NS_OPTIONS(_type, _name) CF_OPTIONS(_type, _name)
#define CF_LANGUAGE_NAME(_name) __attribute__((language_name(#_name)))
#define NS_LANGUAGE_NAME(_name) CF_LANGUAGE_NAME(_name)

typedef NS_ENUM(unsigned, SKFuelKind) { SKFuelKindH2, SKFuelKindCH4, SKFuelKindC12H26 };
unsigned SKFuelKindIsCryogenic(SKFuelKind kind) NS_LANGUAGE_NAME(getter:SKFuelKind.isCryogenic(self:));
unsigned SKFuelKindIsNotCryogenic(SKFuelKind kind) NS_LANGUAGE_NAME(getter:SKFuelKind.isNotCryogenic(self:));
