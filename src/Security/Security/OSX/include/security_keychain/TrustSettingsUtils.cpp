/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
/*
 * TrustSettingsUtils.cpp - Utility routines for TrustSettings module
 *
 */

#include "TrustSettingsUtils.h"
#include <Security/cssmtype.h>
#include <Security/cssmapple.h>
#include <Security/oidscert.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/fcntl.h>

/* 
 * Preferred location for user root store is ~/Library/Keychain/UserRootCerts.keychain. 
 * If we're creating a root store and there is a file there we iterate thru  
 * ~/Library/Keychains/UserRootCerts_N.keychain, 0 <= N <= 10.
 */
#define kSecUserRootStoreBase			"~/Library/Keychains/UserRootCerts"
#define kSecUserRootStoreExtension		".keychain"

namespace Security {

namespace KeychainCore {

/*
 * Read entire file. 
 */
int tsReadFile(
	const char		*fileName,
	Allocator		&alloc,
	CSSM_DATA		&fileData)		// mallocd via alloc and RETURNED
{
	int rtn;
	int fd;
	struct stat	sb;
	unsigned size;
	
	fileData.Data = NULL;
	fileData.Length = 0;
	fd = open(fileName, O_RDONLY, 0);
	if(fd < 0) {
		return errno;
	}
	rtn = fstat(fd, &sb);
	if(rtn) {
		goto errOut;
	}
	size = (unsigned)sb.st_size;
	fileData.Data = (uint8 *)alloc.malloc(size);
	if(fileData.Data == NULL) {
		rtn = ENOMEM;
		goto errOut;
	}
	rtn = (int)lseek(fd, 0, SEEK_SET);
	if(rtn < 0) {
		goto errOut;
	}
	rtn = (int)read(fd, fileData.Data, (size_t)size);
	if(rtn != (int)size) {
		rtn = EIO;
	}
	else {
		rtn = 0;
		fileData.Length = size;
	}
errOut:
	close(fd);
	return rtn;
}

} /* end namespace KeychainCore */

} /* end namespace Security */
