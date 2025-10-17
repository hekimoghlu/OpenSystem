/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#ifndef _LIBINFO_MUSER_H_
#define _LIBINFO_MUSER_H_

/*
 * SPI defining the interface between the libinfo "muser" module and
 * the process providing the service.
 */

#define kLibinfoMultiuserPortName "com.apple.system.libinfo.muser"

/* Request type: one of the kLIMMessageRequest* defines below */
#define kLIMMessageReqtype "reqtype"
/* Available: query whether the muser system is active */
#define kLIMMessageAvailable "available"
/* Query: request dependent query item */
#define kLIMMessageQuery "query"
/* Name: the name of the calling libinfo function */
#define kLIMMessageRPCName "procname"
/* Version: the API version of this request */
#define kLIMMessageVersion "version"

/* 
 * Request a passwd structure for a given username, query type should be
 * an string value with the requested username.
 */
#define kLIMMessageRequestUsername "username"
#define kLIMMessageRequestUID "uid"
#define kLIMMessageRequestGroupname "groupname"
#define kLIMMessageRequestGID "gid"
#define kLIMMessageRequestGrouplist "grouplist"

/* Reply keys from the availability query. */
#define kLIMMessageReplyAvailable "available" // bool

/* Reply keys from user queries. */
#define kLIMMessageReplyName "pw_name" // string
#define kLIMMessageReplyPasswd "pw_passwd" // string
#define kLIMMessageReplyUID "pw_uid" // int64
#define kLIMMessageReplyGID "pw_gid" // int64
#define kLIMMessageReplyHome "pw_dir" // string
#define kLIMMessageReplyShell "pw_shell" // string

/* Reply keys from group queries. */
#define kLIMMessageReplyGroupname "gr_name" // string
#define kLIMMessageReplyGroupID "gr_gid" // int64
#define kLIMMessageReplyGroupMembers "gr_members" // array of strings

/* Reply keys from grouplist queries. */
#define kLIMMessageReplyGrouplist "grouplist" // array of int64

#endif /* _LIBINFO_MUSER_H_ */
