/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
// constdata - shared constant binary data objects
//
#ifndef _H_CONSTDATA
#define _H_CONSTDATA

#include <security_utilities/utilities.h>
#include <security_utilities/refcount.h>


namespace Security {


//
// ConstData represents a contiguous, binary blob of constant data.
// Assignment is by sharing (thus cheap).
// ConstData is a (constant) Dataoid type.
//
class ConstData {    
private:
    class Blob : public RefCount {
    public:
        Blob() : mData(NULL), mSize(0) { }
        Blob(const void *base, size_t size, bool takeOwnership = false);
        ~Blob()		{ delete[] reinterpret_cast<const char *>(mData); }
    
        const void *data() const	{ return mData; }
        size_t length() const		{ return mSize; }
    
    private:
        const void *mData;
        size_t mSize;
    };
    
public:
    ConstData() { }		//@@@ use a nullBlob?
    ConstData(const void *base, size_t size, bool takeOwnership = false)
        : mBlob(new Blob(base, size, takeOwnership)) { }
        
    template <class T>
    static ConstData wrap(const T &obj, bool takeOwnership)
    { return ConstData(&obj, sizeof(obj), takeOwnership); }
    
public:
    const void *data() const	{ return mBlob ? mBlob->data() : NULL; }
    size_t length() const		{ return mBlob ? mBlob->length() : 0; }
    
    operator bool() const		{ return mBlob; }
    bool operator !() const		{ return !mBlob; }

    template <class T> operator const T *() const
    { return reinterpret_cast<const T *>(data()); }
    
    template <class T> const T &as() const
    { return *static_cast<const T *>(reinterpret_cast<const T *>(data())); }
    
private:
    RefPointer<Blob> mBlob;
};


}	// end namespace Security


#endif //_H_CONSTDATA
