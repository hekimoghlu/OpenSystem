/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
/* This is included by cleanup.c multiple times, once for every segement in
 * the _exit_cleanup() code.  This produces the next "case N:" statement in
 * sequence and increments the cleanup_step variable by 1.  This ensures that
 * our case statements never get out of whack due to added/removed steps. */

#if !defined EXIT_CLEANUP_CASE_0
#define EXIT_CLEANUP_CASE_0
	case 0:
#elif !defined EXIT_CLEANUP_CASE_1
#define EXIT_CLEANUP_CASE_1
	case 1:
#elif !defined EXIT_CLEANUP_CASE_2
#define EXIT_CLEANUP_CASE_2
	case 2:
#elif !defined EXIT_CLEANUP_CASE_3
#define EXIT_CLEANUP_CASE_3
	case 3:
#elif !defined EXIT_CLEANUP_CASE_4
#define EXIT_CLEANUP_CASE_4
	case 4:
#elif !defined EXIT_CLEANUP_CASE_5
#define EXIT_CLEANUP_CASE_5
	case 5:
#elif !defined EXIT_CLEANUP_CASE_6
#define EXIT_CLEANUP_CASE_6
	case 6:
#elif !defined EXIT_CLEANUP_CASE_7
#define EXIT_CLEANUP_CASE_7
	case 7:
#elif !defined EXIT_CLEANUP_CASE_8
#define EXIT_CLEANUP_CASE_8
	case 8:
#elif !defined EXIT_CLEANUP_CASE_9
#define EXIT_CLEANUP_CASE_9
	case 9:
#elif !defined EXIT_CLEANUP_CASE_10
#define EXIT_CLEANUP_CASE_10
	case 10:
#elif !defined EXIT_CLEANUP_CASE_11
#define EXIT_CLEANUP_CASE_11
	case 11:
#elif !defined EXIT_CLEANUP_CASE_12
#define EXIT_CLEANUP_CASE_12
	case 12:
#elif !defined EXIT_CLEANUP_CASE_13
#define EXIT_CLEANUP_CASE_13
	case 13:
#elif !defined EXIT_CLEANUP_CASE_14
#define EXIT_CLEANUP_CASE_14
	case 14:
#elif !defined EXIT_CLEANUP_CASE_15
#define EXIT_CLEANUP_CASE_15
	case 15:
#elif !defined EXIT_CLEANUP_CASE_16
#define EXIT_CLEANUP_CASE_16
	case 16:
#else
#error Need to add more case statements!
#endif
		cleanup_step++;
