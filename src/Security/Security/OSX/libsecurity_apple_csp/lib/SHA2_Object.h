/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
 * SHA2_Object.h - SHA2 digest objects 
 *
 */

#ifndef	_SHA2_OBJECT_H_
#define _SHA2_OBJECT_H_

#include <security_cdsa_utilities/digestobject.h>
#include <CommonCrypto/CommonDigest.h>

class SHA224Object : public DigestObject
{
public:
	SHA224Object() { }
	virtual ~SHA224Object() { };
	virtual void digestInit();
	virtual void digestUpdate(
		const void 	*data, 
		size_t 		len);
	virtual void digestFinal(
		void 		*digest);
	virtual DigestObject *digestClone() const;
	virtual size_t digestSizeInBytes() const;
private:
	CC_SHA256_CTX		mCtx;
};

class SHA256Object : public DigestObject
{
public:
	SHA256Object() { }
	virtual ~SHA256Object() { };
	virtual void digestInit();
	virtual void digestUpdate(
		const void 	*data, 
		size_t 		len);
	virtual void digestFinal(
		void 		*digest);
	virtual DigestObject *digestClone() const;
	virtual size_t digestSizeInBytes() const;
private:
	CC_SHA256_CTX		mCtx;
};

class SHA384Object : public DigestObject
{
public:
	SHA384Object() { }
	virtual ~SHA384Object() { };
	virtual void digestInit();
	virtual void digestUpdate(
		const void 	*data, 
		size_t 		len);
	virtual void digestFinal(
		void 		*digest);
	virtual DigestObject *digestClone() const;
	virtual size_t digestSizeInBytes() const;
private:
	CC_SHA512_CTX		mCtx;
};

class SHA512Object : public DigestObject
{
public:
	SHA512Object() { }
	virtual ~SHA512Object() { };
	virtual void digestInit();
	virtual void digestUpdate(
		const void 	*data, 
		size_t 		len);
	virtual void digestFinal(
		void 		*digest);
	virtual DigestObject *digestClone() const;
	virtual size_t digestSizeInBytes() const;
private:
	CC_SHA512_CTX		mCtx;
};

#endif	/* _SHA2_OBJECT_H_ */
