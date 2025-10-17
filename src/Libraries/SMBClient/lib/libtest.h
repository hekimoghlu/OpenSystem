/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#define	MAX_NUMBER_TESTSTRING 6
#define TESTSTRING1 "/George/J\\im/T:?.*est"
#define TESTSTRING2 "/George/J\\im/T:?.*est/"
#define TESTSTRING3 "George/J\\im/T:?.*est/"
#define TESTSTRING4 "George/J\\im/T:?.*est"
#define TESTSTRING5 "George"
#define TESTSTRING6 "/"
#define TESTSTRING7 "/George"

#define	MAX_URL_TO_DICT_TO_URL_TEST			4
#define URL_TO_DICT_TO_URL_TEST_STR1 "smb://local1:local@[fe80::d9b6:f149:a17c:8307%25en1]/Vista-Share"
#define URL_TO_DICT_TO_URL_TEST_STR2 "smb://local:local@colley%5B3%5D._smb._tcp.local/local"
#define URL_TO_DICT_TO_URL_TEST_STR3 "smb://BAD%3A%3B%2FBAD@colley2/badbad"
#define URL_TO_DICT_TO_URL_TEST_STR4 "smb://user:password@%7e%21%40%24%3b%27%28%29/share"


#define LIST_SHARE_CONNECT_TEST				0
#define MOUNT_WIN2003_VOLUME_TEST			1
#define URL_TO_DICT_TO_URL_TEST				3
#define DFS_MOUNT_TEST						6
#define	DFS_LOOP_TEST						7
#define URL_TO_DICTIONARY					8
#define FIND_SESSION_FROM_MP_TEST			9
#define NETFS_TEST							10
#define GETACCOUNTNAME_AND_SID_TEST			11
#define FORCE_GUEST_ANON_TEST				12
#define MOUNT_EXIST_TEST					13
#define LIST_DFS_REFERRALS					14
#define RUN_ALL_TEST						-1

#define START_UNIT_TEST		LIST_SHARE_CONNECT_TEST
#define END_UNIT_TEST		LIST_DFS_REFERRALS
/* Should always be greater than END_UNIT_TEST */
#define REMOUNT_UNIT_TEST	END_UNIT_TEST+1

