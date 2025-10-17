/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#pragma once

#include "stdafx.h"
#include "resource.h"

//---------------------------------------------------------------------------------------------------------------------------
//	CConfigDialog
//---------------------------------------------------------------------------------------------------------------------------

class CConfigDialog : public CDialog
{
public:

	CConfigDialog();

protected:

	//{{AFX_DATA(CConfigDialog)
	enum { IDD = IDR_APPLET };
	//}}AFX_DATA

	//{{AFX_VIRTUAL(CConfigDialog)
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

	//{{AFX_MSG(CConfigDialog)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

	DECLARE_DYNCREATE(CConfigDialog)
};
