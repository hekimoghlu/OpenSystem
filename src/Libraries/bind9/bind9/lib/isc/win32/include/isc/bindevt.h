/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
/* $Id: bindevt.h,v 1.6 2007/06/19 23:47:20 tbox Exp $ */

#ifndef ISC_BINDEVT_H
#define ISC_BINDEVT_H 1

/*
 * This is used for the event log for both logging the messages and
 * later on by the event viewer when looking at the events
 */

/*
 * Values are 32 bit values layed out as follows:
 *
 *   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
 *   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
 *  +---+-+-+-----------------------+-------------------------------+
 *  |Sev|C|R|     Facility          |               Code            |
 *  +---+-+-+-----------------------+-------------------------------+
 *
 *  where
 *
 *      Sev - is the severity code
 *
 *          00 - Success
 *          01 - Informational
 *          10 - Warning
 *          11 - Error
 *
 *      C - is the Customer code flag
 *
 *      R - is a reserved bit
 *
 *      Facility - is the facility code
 *
 *      Code - is the facility's status code
 *
 *
 * Define the facility codes
 */
 

/*
 * Define the severity codes
 */


/*
 * MessageId: BIND_ERR_MSG
 *
 * MessageText:
 *
 *  %1
 */
#define BIND_ERR_MSG		((DWORD)0xC0000001L)

/*
 * MessageId: BIND_WARN_MSG
 *
 * MessageText:
 *
 *  %1
 */
#define BIND_WARN_MSG		((DWORD)0x80000002L)

/*
 * MessageId: BIND_INFO_MSG
 *
 * MessageText:
 *
 *  %1
 */
#define BIND_INFO_MSG		((DWORD)0x40000003L)

#endif /* ISC_BINDEVT_H */
