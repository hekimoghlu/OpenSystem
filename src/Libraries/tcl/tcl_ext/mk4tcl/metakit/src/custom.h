/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

// custom.h --
// $Id: custom.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, the homepage is http://www.equi4.com/metakit.html

/** @file
 * Encapsulation of many custom viewer classes
 */

#ifndef __CUSTOM_H__
#define __CUSTOM_H__

#ifndef __FIELD_H__
#include "field.h"
#endif 
#ifndef __STORE_H__
#include "handler.h"
#endif 

/////////////////////////////////////////////////////////////////////////////

class c4_CustomSeq: public c4_HandlerSeq {
    c4_CustomViewer *_viewer;
    bool _inited;

  public:
    c4_CustomSeq(c4_CustomViewer *viewer_);
    virtual ~c4_CustomSeq();

    virtual int NumRows()const;

    virtual bool RestrictSearch(c4_Cursor, int &, int &);

    virtual void InsertAt(int, c4_Cursor, int = 1);
    virtual void RemoveAt(int, int = 1);
    virtual void Move(int from_, int);

    bool DoGet(int row_, int col_, c4_Bytes &buf_)const;
    void DoSet(int row_, int col_, const c4_Bytes &buf_);

  private:
    // this *is* used, as override
    virtual c4_Handler *CreateHandler(const c4_Property &);
};

/////////////////////////////////////////////////////////////////////////////

extern c4_CustomViewer *f4_CustSlice(c4_Sequence &, int, int, int);
extern c4_CustomViewer *f4_CustProduct(c4_Sequence &, const c4_View &);
extern c4_CustomViewer *f4_CustRemapWith(c4_Sequence &, const c4_View &);
extern c4_CustomViewer *f4_CustPair(c4_Sequence &, const c4_View &);
extern c4_CustomViewer *f4_CustConcat(c4_Sequence &, const c4_View &);
extern c4_CustomViewer *f4_CustRename(c4_Sequence &, const c4_Property &, const
  c4_Property &);
extern c4_CustomViewer *f4_CustGroupBy(c4_Sequence &, const c4_View &, const
  c4_Property &);
extern c4_CustomViewer *f4_CustJoinProp(c4_Sequence &, const c4_ViewProp &,
  bool);
extern c4_CustomViewer *f4_CustJoin(c4_Sequence &, const c4_View &, const
  c4_View &, bool);

/////////////////////////////////////////////////////////////////////////////

#endif
