/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

//---------------------------------------------------------------------------------------------------------------------------
//	CCPApplet
//---------------------------------------------------------------------------------------------------------------------------

class CCPApplet : public CCmdTarget
{
public:

	CCPApplet( UINT nResourceID, UINT nDescriptionID, CRuntimeClass* pUIClass );

	virtual ~CCPApplet();

protected:

	virtual LRESULT OnRun(CWnd* pParentWnd);
	virtual LRESULT OnStartParms(CWnd* pParentWnd, LPCTSTR lpszExtra);
	virtual LRESULT OnInquire(CPLINFO* pInfo);
	virtual LRESULT OnNewInquire(NEWCPLINFO* pInfo);
	virtual LRESULT OnSelect();
	virtual LRESULT OnStop();

	CRuntimeClass	*	m_uiClass;
	UINT				m_resourceId;
	UINT				m_descId;
	CString				m_name;
	int					m_pageNumber;
  
	friend class CCPApp;

	DECLARE_DYNAMIC(CCPApplet);
};


//---------------------------------------------------------------------------------------------------------------------------
//	CCPApp
//---------------------------------------------------------------------------------------------------------------------------

class CCPApp : public CWinApp
{
public:

	CCPApp();
	virtual ~CCPApp();

	void AddApplet( CCPApplet* pApplet );

protected:

	CList<CCPApplet*, CCPApplet*&> m_applets;

	friend LONG APIENTRY
	CPlApplet(HWND hWndCPl, UINT uMsg, LONG lParam1, LONG lParam2);

	virtual LRESULT OnCplMsg(HWND hWndCPl, UINT msg, LPARAM lp1, LPARAM lp2);
	virtual LRESULT OnInit();
	virtual LRESULT OnExit();

	DECLARE_DYNAMIC(CCPApp);
};


CCPApp * GetControlPanelApp();
