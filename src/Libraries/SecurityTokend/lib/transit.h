/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef _H_TRANSIT
#define _H_TRANSIT

#include "tdclient.h"
#include "tokend.h"
#include "server.h"
#include <security_cdsa_utilities/cssmwalkers.h>
#include <security_cdsa_utilities/u32handleobject.h>
#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>


namespace Security {
namespace Tokend {


using namespace Security::Tokend;
using namespace Security::DataWalkers;


#define TOKEND_ARGS \
	mach_port_t servicePort, mach_port_t replyPort, CSSM_RETURN *rcode

#define CONTEXT_ARGS Context context, Pointer contextBase, Context::Attr *attributes, mach_msg_type_number_t attrSize

#define BEGIN_IPC	*rcode = CSSM_OK; try {
#define END_IPC(base) } \
	catch (const CommonError &err) { *rcode = CssmError::cssmError(err, CSSM_ ## base ## _BASE_ERROR); } \
	catch (const std::bad_alloc &) { *rcode = CssmError::merge(CSSM_ERRCODE_MEMORY_ERROR, CSSM_ ## base ## _BASE_ERROR); } \
	catch (...) { *rcode = CssmError::cssmError(CSSM_ERRCODE_INTERNAL_ERROR, CSSM_ ## base ## _BASE_ERROR); } \
	return KERN_SUCCESS;

#define DATA_IN(base)	void *base, mach_msg_type_number_t base##Length
#define DATA_OUT(base)	void **base, mach_msg_type_number_t *base##Length
#define DATA(base)		CssmData(base, base##Length)

#define COPY_IN(type,name)	type *name, mach_msg_type_number_t name##Length, type *name##Base
#define COPY_OUT(type,name)	\
	type **name, mach_msg_type_number_t *name##Length, type **name##Base


#define CALL(func,args)	{ \
	if (server->func) \
		{ if (*rcode = server->func args) return KERN_SUCCESS; /* and rcode */ } \
	else \
		CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED); }


using LowLevelMemoryUtilities::increment;
using LowLevelMemoryUtilities::difference;


//
// An OutputData object will take memory allocated within the SecurityServer,
// hand it to the MIG return-output parameters, and schedule it to be released
// after the MIG reply has been sent. It will also get rid of it in case of
// error.
//
class OutputData : public CssmData {
public:
	OutputData(void **outP, mach_msg_type_number_t *outLength)
		: mData(*outP), mLength(*outLength) { }
	~OutputData()
	{ mData = data(); mLength = length(); server->releaseWhenDone(Allocator::standard(), mData); }
    
    void operator = (const CssmData &source)
    { CssmData::operator = (source); }
	
private:
	void * &mData;
	mach_msg_type_number_t &mLength;
};


//
// A local copy of a structured return value (COPY_OUT form), self-managing
//
template <class T>
class Return : public T {
public:
	Return(T **p, mach_msg_type_number_t *l, T **b)
		: ptr(*p), len(*l), base(*b) { ptr = NULL; }
	~Return();

private:
	T *&ptr;
	mach_msg_type_number_t &len;
	T *&base;
};

template <class T>
Return<T>::~Return()
{
	Copier<T> copier(static_cast<T*>(this), Allocator::standard());
	ptr = base = copier;
	len = copier.length();
	server->releaseWhenDone(Allocator::standard(), copier.keep());
}


//
// Relocation support
//
void relocate(Context &context, void *base, Context::Attr *attrs, uint32 attrSize);


//
// Data search and retrieval interface
//
class DataRetrieval : public TOKEND_RETURN_DATA {
public:
	DataRetrieval(CssmDbRecordAttributeData *inAttributes, bool getData);
	~DataRetrieval();
	void returnData(KeyHandle &hKey, RecordHandle &hRecord,
		void *&outData, mach_msg_type_number_t &outDataLength,
		CssmDbRecordAttributeData *&outAttributes, mach_msg_type_number_t &outAttrLength,
		CssmDbRecordAttributeData *&outAttributesBase);
	
private:
	CssmData mData;
};
    
//
// Tokend handles <---> client-IPC handles
// the "key" into the map is the 32 bit handle, the value is 64 bit (i.e. native)
//
class TokendHandle : public TypedHandle<CSSM_HANDLE>
{
public:
    TokendHandle(CSSM_HANDLE cssmh) : TypedHandle<CSSM_HANDLE>(cssmh)  { }
    virtual ~TokendHandle()  { }
};

class TokendHandleObject : public U32HandleObject
{

public:
    // typedef TypedHandle<CSSM_HANDLE> TokendHandle;
	typedef uint32_t TransitHandle;		// maybe should be U32HandleObject::Handle
	
public:
    static TokendHandleObject *make(CSSM_HANDLE cssmh);
	static CSSM_HANDLE findTDHandle(TransitHandle thdl);

    TokendHandleObject(CSSM_HANDLE tokendh);
    virtual ~TokendHandleObject();
    CSSM_HANDLE tokendHandle() const  { return mTokendHandle.handle(); }
    U32HandleObject::Handle ipcHandle() const  { return U32HandleObject::handle(); }

private:
    TokendHandle mTokendHandle;
    
	// key,value
    typedef std::map<TransitHandle, TokendHandleObject *> TokendHandleMap;
    class TokendHandleMapState : public Mutex, public TokendHandleMap
    {
    public:
        TokendHandleMapState();
        bool handleInUse(CSSM_HANDLE cssmh);
        void add(TransitHandle thdl, TokendHandleObject *tho);
		void erase(TransitHandle thdl);
    };
    static ModuleNexus<TokendHandleObject::TokendHandleMapState> thmstate;
};

}	// namespace Tokend
}	// namespace Security

#endif //_H_TRANSIT
