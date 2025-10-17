/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#ifndef WCDataObject_h
#define WCDataObject_h

#include "DragData.h"
#include <ShlObj.h>
#include <objidl.h>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WebCore {

struct StgMediumDeleter {
    void operator()(STGMEDIUM* medium)
    {
        ::ReleaseStgMedium(medium);
    }
};

class WCDataObject final : public IDataObject {
public:
    void CopyMedium(STGMEDIUM* pMedDest, STGMEDIUM* pMedSrc, FORMATETC* pFmtSrc);

    //IUnknown
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(_In_ REFIID riid, _COM_Outptr_ void** ppvObject);        
    virtual ULONG STDMETHODCALLTYPE AddRef();
    virtual ULONG STDMETHODCALLTYPE Release();

    //IDataObject
    virtual HRESULT STDMETHODCALLTYPE GetData(FORMATETC* pformatIn, STGMEDIUM* pmedium);
    virtual HRESULT STDMETHODCALLTYPE GetDataHere(FORMATETC* pformat, STGMEDIUM* pmedium);
    virtual HRESULT STDMETHODCALLTYPE QueryGetData(FORMATETC* pformat);
    virtual HRESULT STDMETHODCALLTYPE GetCanonicalFormatEtc(FORMATETC* pformatectIn,FORMATETC* pformatOut);
    virtual HRESULT STDMETHODCALLTYPE SetData(FORMATETC* pformat, STGMEDIUM*pmedium, BOOL release);
    virtual HRESULT STDMETHODCALLTYPE EnumFormatEtc(DWORD dwDirection, IEnumFORMATETC** ppenumFormatEtc);
    virtual HRESULT STDMETHODCALLTYPE DAdvise(FORMATETC*, DWORD, IAdviseSink*, DWORD*);
    virtual HRESULT STDMETHODCALLTYPE DUnadvise(DWORD);
    virtual HRESULT STDMETHODCALLTYPE EnumDAdvise(IEnumSTATDATA**);

    void clearData(CLIPFORMAT);
    
    static HRESULT createInstance(WCDataObject**);
    static HRESULT createInstance(WCDataObject**, const DragDataMap&);
private:
    WCDataObject();

    Vector<std::unique_ptr<FORMATETC>> m_formats;
    Vector<std::unique_ptr<STGMEDIUM, StgMediumDeleter>> m_medium;
    long m_ref { 1 };
};

}

#endif //!WCDataObject_h
