/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

//
// This file contains message strings for the OpenLDAP slapd service.
//
// This file should be compiled as follows
//   mc -v slapdmsg.mc  -r $(IntDir)  
//   rc /v /r  $(IntDir)\slapdmsg.rc
// The mc (message compiler) command generates the .rc and .h files from this file. The 
// rc (resource compiler) takes the .rc file and produces a .res file that can be linked 
// with the final executable application. The application is then registered as a message
// source with by creating the appropriate entries in the system registry.
//
//
//  Values are 32 bit values layed out as follows:
//
//   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
//   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
//  +---+-+-+-----------------------+-------------------------------+
//  |Sev|C|R|     Facility          |               Code            |
//  +---+-+-+-----------------------+-------------------------------+
//
//  where
//
//      Sev - is the severity code
//
//          00 - Success
//          01 - Informational
//          10 - Warning
//          11 - Error
//
//      C - is the Customer code flag
//
//      R - is a reserved bit
//
//      Facility - is the facility code
//
//      Code - is the facility's status code
//
//
// Define the facility codes
//


//
// Define the severity codes
//


//
// MessageId: MSG_SVC_STARTED
//
// MessageText:
//
//  OpenLDAP service started. debuglevel=%1, conffile=%2, urls=%3
//
#define MSG_SVC_STARTED                  0x40000500L

//
// MessageId: MSG_SVC_STOPPED
//
// MessageText:
//
//  OpenLDAP service stopped.
//
#define MSG_SVC_STOPPED                  0x40000501L

