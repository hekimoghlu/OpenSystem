/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
 * SHA1_MD5_Object.h - SHA1, MD5 digest objects 
 *
 */

#ifndef	_SHA1_MD5_OBJECT_H_
#define _SHA1_MD5_OBJECT_H_

#include <security_cdsa_utilities/digestobject.h>
#include <CommonCrypto/CommonDigest.h>

class SHA1Object : public DigestObject
{
public:
	SHA1Object() { }
	virtual ~SHA1Object() { };
	virtual void digestInit();
	virtual void digestUpdate(
		const void 	*data, 
		size_t 		len);
	virtual void digestFinal(
		void 		*digest);
	virtual DigestObject *digestClone() const;
	virtual size_t digestSizeInBytes() const;
private:
	CC_SHA1_CTX		mCtx;
};

class MD5Object : public DigestObject
{
public:
	MD5Object() { }
	virtual ~MD5Object() { }
	virtual void digestInit();
	virtual void digestUpdate(
		const void 	*data, 
		size_t 		len);
	virtual void digestFinal(
		void 		*digest);
	virtual DigestObject *digestClone() const;
	virtual size_t digestSizeInBytes() const;
private:
	CC_MD5_CTX mCtx;
};

#endif	/* _SHA1_MD5_OBJECT_H_ */
