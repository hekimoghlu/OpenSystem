/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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

#include <wtf/Vector.h>

// FIXME: <rdar://problem/31618371>
// The following constants and functions are for customized DER implementations.
// They are not intended to be used outside Crypto codes, and should be removed
// once the above bug is fixed.
namespace WebCore {

// Per X.690 08/2015: https://www.itu.int/rec/T-REC-X.680-X.693/en
static const unsigned char BitStringMark = 0x03;
static const unsigned char IntegerMark = 0x02;
static const unsigned char OctetStringMark = 0x04;
static const unsigned char SequenceMark = 0x30;
// Version 0. Per https://tools.ietf.org/html/rfc5208#section-5
static const unsigned char Version[] = {0x02, 0x01, 0x00};

static const unsigned char InitialOctet = 0x00;
static const size_t MaxLengthInOneByte = 128;

size_t bytesUsedToEncodedLength(uint8_t);
size_t extraBytesNeededForEncodedLength(size_t);
void addEncodedASN1Length(Vector<uint8_t>&, size_t);
size_t bytesNeededForEncodedLength(size_t);

} // namespace WebCore
