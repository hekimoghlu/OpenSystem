/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
/* $Id: DirBrowse.cpp,v 1.6 2007/06/19 23:47:07 tbox Exp $ */

/*
 * Copyright (c) 1999-2000 by Nortel Networks Corporation
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND NORTEL NETWORKS DISCLAIMS
 * ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NORTEL NETWORKS
 * BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
 * ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

#include "stdafx.h"
#include "BINDInstall.h"
#include "DirBrowse.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CDirBrowse dialog


CDirBrowse::CDirBrowse(CString initialDir, CWnd* pParent /*=NULL*/)
	: CDialog(CDirBrowse::IDD, pParent)
{
	//{{AFX_DATA_INIT(CDirBrowse)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
	m_selectedDir = initialDir;
}


void CDirBrowse::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CDirBrowse)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CDirBrowse, CDialog)
	//{{AFX_MSG_MAP(CDirBrowse)
	ON_LBN_DBLCLK(IDC_DIRLIST, OnDblclkDirlist)
	ON_LBN_SELCHANGE(IDC_DIRLIST, OnSelchangeDirlist)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CDirBrowse message handlers

BOOL CDirBrowse::OnInitDialog() 
{
	CDialog::OnInitDialog();
	
	DlgDirList((LPTSTR)(LPCTSTR)m_selectedDir, IDC_DIRLIST, IDC_CURDIR, DDL_DIRECTORY);
	
	return TRUE;  // return TRUE unless you set the focus to a control
	              // EXCEPTION: OCX Property Pages should return FALSE
}

void CDirBrowse::OnDblclkDirlist() 
{
	CListBox *lb = (CListBox *)GetDlgItem(IDC_DIRLIST);
	CString curSel;

	lb->GetText(lb->GetCurSel(), curSel);
	DlgDirList((LPTSTR)(LPCTSTR)curSel, IDC_DIRLIST, IDC_CURDIR, DDL_DIRECTORY);
}

void CDirBrowse::OnSelchangeDirlist() 
{
	// TODO: Add your control notification handler code here
	
}
