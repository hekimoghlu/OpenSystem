/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "stdafx.h"
#include "PrinterSetupWizardApp.h"
#include "PrinterSetupWizardSheet.h"
#include "FirstPage.h"

#include <DebugServices.h>


// CFirstPage dialog

IMPLEMENT_DYNAMIC(CFirstPage, CPropertyPage)
CFirstPage::CFirstPage()
	: CPropertyPage(CFirstPage::IDD)
{
	CString fontName;

	m_psp.dwFlags &= ~(PSP_HASHELP);
	m_psp.dwFlags |= PSP_DEFAULT|PSP_HIDEHEADER;

	fontName.LoadString(IDS_LARGE_FONT);

	// create the large font
	m_largeFont.CreateFont(-16, 0, 0, 0, 
		FW_BOLD, FALSE, FALSE, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, 
		CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH, fontName);
}

CFirstPage::~CFirstPage()
{
}

void CFirstPage::DoDataExchange(CDataExchange* pDX)
{
	CPropertyPage::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_GREETING, m_greeting);
}


BOOL
CFirstPage::OnSetActive()
{
	CPrinterSetupWizardSheet * psheet;
	CString greetingText;

	psheet = reinterpret_cast<CPrinterSetupWizardSheet*>(GetParent());
	require_quiet( psheet, exit );   
   
	psheet->SetWizardButtons(PSWIZB_NEXT);

	m_greeting.SetFont(&m_largeFont);

	greetingText.LoadString(IDS_GREETING);
	m_greeting.SetWindowText(greetingText);

exit:

	return CPropertyPage::OnSetActive();
}


BOOL
CFirstPage::OnKillActive()
{
	CPrinterSetupWizardSheet * psheet;

	psheet = reinterpret_cast<CPrinterSetupWizardSheet*>(GetParent());
	require_quiet( psheet, exit );   
   
	psheet->SetLastPage(this);

exit:

	return CPropertyPage::OnKillActive();
}


BEGIN_MESSAGE_MAP(CFirstPage, CPropertyPage)
END_MESSAGE_MAP()


// CFirstPage message handlers
