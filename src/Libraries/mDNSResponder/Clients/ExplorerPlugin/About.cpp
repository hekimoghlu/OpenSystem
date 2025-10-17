/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

// About.cpp : implementation file
//

#include "stdafx.h"
#include "ExplorerPlugin.h"
#include "About.h"
#include "WinVersRes.h"
#include <DebugServices.h>


// CAbout dialog

IMPLEMENT_DYNAMIC(CAbout, CDialog)
CAbout::CAbout(CWnd* pParent /*=NULL*/)
	: CDialog(CAbout::IDD, pParent)
{
	// Initialize brush with the desired background color
	m_bkBrush.CreateSolidBrush(RGB(255, 255, 255));
}

CAbout::~CAbout()
{
}

void CAbout::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMPONENT, m_componentCtrl);
	DDX_Control(pDX, IDC_LEGAL, m_legalCtrl);
}


BEGIN_MESSAGE_MAP(CAbout, CDialog)
ON_WM_CTLCOLOR()
END_MESSAGE_MAP()


// CAbout message handlers
BOOL
CAbout::OnInitDialog()
{
	BOOL b = CDialog::OnInitDialog();

	CStatic * control = (CStatic*) GetDlgItem( IDC_ABOUT_BACKGROUND );
	check( control );

	if ( control )
	{
		control->SetBitmap( ::LoadBitmap( GetNonLocalizedResources(), MAKEINTRESOURCE( IDB_ABOUT ) ) );
	}

	control = ( CStatic* ) GetDlgItem( IDC_COMPONENT_VERSION );
	check( control );

	if ( control )
	{
		control->SetWindowText( TEXT( MASTER_PROD_VERS_STR2 ) );
	}

	return b;
}


HBRUSH CAbout::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{ 
	switch (nCtlColor)
	{
		case CTLCOLOR_STATIC:
	
			if ( pWnd->GetDlgCtrlID() == IDC_COMPONENT )
			{
				pDC->SetTextColor(RGB(64, 64, 64));
			}
			else
			{
				pDC->SetTextColor(RGB(0, 0, 0));
			}

			pDC->SetBkColor(RGB(255, 255, 255));
			return (HBRUSH)(m_bkBrush.GetSafeHandle());

		case CTLCOLOR_DLG:
	
			return (HBRUSH)(m_bkBrush.GetSafeHandle());

		default:
	
			return CDialog::OnCtlColor(pDC, pWnd, nCtlColor);
	}
}
