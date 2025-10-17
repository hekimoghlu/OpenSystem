/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include <darwintest.h>
#include <darwintest_utils.h>
#include <darwintest_multiprocess.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/attr.h>
#include <sys/event.h>
#include <sys/resource.h>
#include <sys/mount.h>

#ifndef ATTR_VOL_FSTYPENAME
#define ATTR_VOL_FSTYPENAME 0x00100000
#endif

#ifndef ATTR_VOL_FSSUBTYPE
#define ATTR_VOL_FSSUBTYPE 0x00200000
#endif

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vfs"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("vfs"),
	T_META_ASROOT(false),
	T_META_CHECK_LEAKS(false));

T_DECL(getattrlist_fstype,
    "test ATTR_VOL_FSTYPENAME and ATTR_VOL_FSSUBTYPE",
    T_META_ASROOT(false))
{
	struct myattrbuf {
		uint32_t length;
		attribute_set_t returned_attrs;
		vol_attributes_attr_t vol_attributes;
		attrreference_t fstypename_ref;
		uint32_t fssubtype;
		char fstypename[MFSTYPENAMELEN];
	} attrbuf;
	struct attrlist attrs = {
		.bitmapcount = ATTR_BIT_MAP_COUNT,
		.commonattr = ATTR_CMN_RETURNED_ATTRS,
		/*
		 * Request ATTR_VOL_ATTRIBUTES to ensure that
		 * ATTR_VOL_FSTYPENAME and ATTR_VOL_FSSUBTYPE
		 * are packed into the buffer *after*.
		 */
		.volattr = ATTR_VOL_INFO | ATTR_VOL_ATTRIBUTES |
	    ATTR_VOL_FSTYPENAME | ATTR_VOL_FSSUBTYPE,
	};
	const char *tmpdir = dt_tmpdir();
	struct statfs sfs;

	T_SETUPBEGIN;

	T_WITH_ERRNO;
	T_ASSERT_POSIX_ZERO(statfs(tmpdir, &sfs),
	    "Setup: statfs'ing tmpdir: %s", tmpdir);

	T_SETUPEND;

	memset(&attrbuf, 0, sizeof(attrbuf));
	T_WITH_ERRNO;
	T_ASSERT_POSIX_ZERO(getattrlist(tmpdir, &attrs, &attrbuf,
	    sizeof(attrbuf), FSOPT_REPORT_FULLSIZE | FSOPT_PACK_INVAL_ATTRS),
	    "Calling getattrlist on tmpdir: %s", tmpdir);

	T_ASSERT_TRUE(attrbuf.length <= sizeof(attrbuf),
	    "Asserting attrbuf.length <= sizeof(attrbuf)");
	T_ASSERT_TRUE(attrbuf.returned_attrs.volattr & ATTR_VOL_FSTYPENAME,
	    "Asserting ATTR_VOL_FSTYPENAME was returned");
	T_ASSERT_TRUE(attrbuf.returned_attrs.volattr & ATTR_VOL_FSSUBTYPE,
	    "Asserting ATTR_VOL_FSSUBTYPE was returned");
	T_ASSERT_TRUE(attrbuf.fstypename_ref.attr_dataoffset + offsetof(struct myattrbuf, fstypename_ref) == offsetof(struct myattrbuf, fstypename),
	    "Asserting fstypename in expected place (%d + %lu == %lu)",
	    attrbuf.fstypename_ref.attr_dataoffset,
	    offsetof(struct myattrbuf, fstypename_ref),
	    offsetof(struct myattrbuf, fstypename));
	T_ASSERT_TRUE(attrbuf.fstypename_ref.attr_length <= MFSTYPENAMELEN,
	    "Asserting fstypename length is valid");
	T_ASSERT_TRUE(strcmp(attrbuf.fstypename, sfs.f_fstypename) == 0,
	    "Asserting that fstypename matches");
	T_ASSERT_TRUE(attrbuf.fssubtype == sfs.f_fssubtype,
	    "Asserting that fssubtype matches");
}
