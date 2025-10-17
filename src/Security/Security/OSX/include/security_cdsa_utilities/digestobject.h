/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
 * digestobject.h - generic virtual Digest base class 
 */

#ifndef	_DIGEST_OBJECT_H_
#define _DIGEST_OBJECT_H_

#include <security_cdsa_utilities/cssmalloc.h>

/* common virtual digest class */
class DigestObject {
public:
	DigestObject() : mInitFlag(false), mIsDone(false) { }
	virtual ~DigestObject() { }
	
	/* 
	 * The remaining functions must be implemented by subclass. 
	 */
	/* init is reusable */
	virtual void digestInit() = 0;

	/* add some data */
	virtual void digestUpdate(
		const void *data, 
		size_t 		len) = 0;
	
	/* obtain digest (once only per init, update, ... cycle) */
	virtual void digestFinal(
		void 		*digest) = 0;  	/* RETURNED, alloc'd by caller */
	
	/* sublass-specific copy */
	virtual DigestObject *digestClone() const = 0;
	
	virtual size_t digestSizeInBytes() const = 0;

protected:
	bool			mInitFlag;
	bool			mIsDone;	
			
	bool			initFlag() 				{ return mInitFlag; }
	void			setInitFlag(bool flag) 	{ mInitFlag = flag; }
	bool			isDone() 				{ return mIsDone; }
	void			setIsDone(bool done) 	{ mIsDone = done; }
};

/*
 * NullDigest.h - nop digest for use with raw signature algorithms.
 *				  NullDigest(someData) = someData.
 */
class NullDigest : public DigestObject
{
public:
	NullDigest() : mInBuf(NULL), mInBufSize(0) 
	{ 
	}

	void digestInit() 
	{ 
		/* reusable - reset */
		if(mInBufSize) {
			assert(mInBuf != NULL);
			memset(mInBuf, 0, mInBufSize);
			Allocator::standard().free(mInBuf);
			mInBufSize = 0;
			mInBuf = NULL;
		}
	}

	~NullDigest()
	{
		digestInit();
	}

	void digestUpdate(
		const void *data, 
		size_t 		len) 
	{
		mInBuf = Allocator::standard().realloc(mInBuf, mInBufSize + len);
		memmove((uint8 *)mInBuf + mInBufSize, data, len);
		mInBufSize += len;
	}
	
	virtual void digestFinal(
		void 		*digest)
	{
		memmove(digest, mInBuf, mInBufSize);
	}
										
	virtual DigestObject *digestClone() const
	{
		NullDigest *cloned = new NullDigest;
		cloned->digestUpdate(mInBuf, mInBufSize);
		return cloned;
	}
	
	/* unique to NullDigest - just obtain current data ptr, no copy */
	virtual const void *digestPtr() { return mInBuf; }
	
	size_t digestSizeInBytes() const
	{ 
		return mInBufSize;
	}

private:
	void		*mInBuf;
	size_t		mInBufSize;
};

#endif	/* _DIGEST_OBJECT_H_ */
