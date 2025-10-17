/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#ifndef __DECOMPDATAENUMS__
#define __DECOMPDATAENUMS__

// Basic table parameters for 2-stage trie:
// The high 12 bits of a UniChar provide an index into a first-level table;
// if the entry there is >= 0, it is an index into a table of 16-element
// ranges indexed by the low 4 bits of the UniChar. Since the UniChars of interest
// for combining classes and sequence updates are either in the range 0000-30FF
// or in the range FB00-FFFF, we eliminate the large middle section of the first-
// level table by first adding 0500 to the UniChar to wrap the UniChars of interest
// into the range 0000-35FF.
enum {
	kLoFieldBitSize		= 4,
	kShiftUniCharOffset	= 0x0500,	// add to UniChar so FB00 & up wraps to 0000
	kShiftUniCharLimit	= 0x3600	// if UniChar + offset >= limit, no need to check
};

// The following are all derived from kLoFieldBitSize
enum {
	kLoFieldEntryCount	= 1 << kLoFieldBitSize,
	kHiFieldEntryCount	= kShiftUniCharLimit >> kLoFieldBitSize,
	kLoFieldMask		= (1 << kLoFieldBitSize) - 1
};

// Action codes for sequence replacement/updating
enum {											// next + repl = total chars
	// a value of 0 means no action
	kReplaceCurWithTwo					= 0x02,	//    0 + 2 = 2
	kReplaceCurWithThree				= 0x03,	//    0 + 3 = 3
	kIfNextOneMatchesReplaceAllWithOne	= 0x12,	//    1 + 1 = 2
	kIfNextOneMatchesReplaceAllWithTwo	= 0x13,	//    1 + 2 = 3
	kIfNextTwoMatchReplaceAllWithOne	= 0x23	//    2 + 1 = 3
};

#endif // __FSCKFIXDECOMPS__


