/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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

enum class ScopedEnumDefined { x = 0, y = 2 };

enum class ScopedEnumBasic { x, y, z };

enum class ScopedEnumCharDefined : char { x = 0, y = 2 };

enum class ScopedEnumUnsignedDefined : unsigned int { x = 0, y = 2 };

enum class ScopedEnumUnsignedLongDefined : unsigned long { x = 0, y = 2 };

enum class ScopedEnumChar : char { x, y, z };

enum class ScopedEnumUnsigned : unsigned int { x, y, z };

enum class ScopedEnumUnsignedLong : unsigned long { x, y, z };

enum class ScopedEnumInt : int { x, y, z };

enum class ScopedEnumNegativeElement : int { x = -1, y = 0, z = 2 };

enum class MiddleDefinedScopedEnum { x, y = 42, z };
