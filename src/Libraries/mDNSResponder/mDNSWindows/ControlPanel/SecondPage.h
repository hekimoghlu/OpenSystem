/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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

#include <DebugServices.h>
#include <list>


//---------------------------------------------------------------------------------------------------------------------------
//	CSecondPage
//---------------------------------------------------------------------------------------------------------------------------

class CSecondPage : public CPropertyPage
{
public:
	CSecondPage();
	~CSecondPage();

protected:

	//{{AFX_DATA(CSecondPage)
	enum { IDD = IDR_APPLET_PAGE2 };
	//}}AFX_DATA

	//{{AFX_VIRTUAL(CSecondPage)
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

	DECLARE_DYNCREATE(CSecondPage)

	//{{AFX_MSG(CSecondPage)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
	
	afx_msg void	OnBnClickedSharedSecret();
	afx_msg void	OnBnClickedAdvertise();

	void			OnAddRegistrationDomain( CString & domain );
	void			OnRemoveRegistrationDomain( CString & domain );
	
private:
	
	typedef std::list<CString> StringList;

	afx_msg BOOL
	OnSetActive();
	
	afx_msg void
	OnOK();

	void
	EmptyComboBox
			(
			CComboBox	&	box
			);

	OSStatus
	Populate(
			CComboBox	&	box,
			HKEY			key,
			StringList	&	l
			);
	
	void
	SetModified( BOOL bChanged = TRUE );
	
	void
	Commit();

	OSStatus
	Commit( CComboBox & box, HKEY key, DWORD enabled );

	OSStatus
	CreateKey( CString & name, DWORD enabled );

	OSStatus
	RegQueryString( HKEY key, CString valueName, CString & value );

	CComboBox		m_regDomainsBox;
	CButton			m_advertiseServicesButton;
	CButton			m_sharedSecretButton;
	BOOL			m_modified;
	HKEY			m_setupKey;

public:
	afx_msg void OnCbnSelChange();
	afx_msg void OnCbnEditChange();
}; 
