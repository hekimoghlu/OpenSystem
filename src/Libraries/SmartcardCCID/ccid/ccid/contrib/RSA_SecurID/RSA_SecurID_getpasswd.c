/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winscard.h>

/* DWORD printf(3) format */
#ifdef __APPLE__
/* Apple defines DWORD as uint32_t so %d is correct */
#define LF
#else
/* pcsc-lite defines DWORD as unsigned long so %ld is correct */
#define LF "l"
#endif

/* PCSC error message pretty print */
#define PCSC_ERROR_EXIT(rv, text) \
if (rv != SCARD_S_SUCCESS) \
{ \
	printf(text ": %s (0x%"LF"X)\n", pcsc_stringify_error(rv), rv); \
	goto end; \
}

int main(void)
{
	unsigned char cmd1[] = { 0x00, 0xa4, 0x04, 0x00, 0x0a, 0xa0, 0x00, 0x00, 0x00, 0x63, 0x86, 0x53, 0x49, 0x44, 0x01};
	unsigned char cmd2[] = { 0x80, 0x56, 0x00, 0x00, 0x04 };
	unsigned char cmd3[] = { 0x80, 0x48, 0x00, 0x00, 0x04, 0xff, 0xff, 0xff, 0xff };
	unsigned char cmd4[] = { 0x80, 0x44, 0x00, 0x00, 0x05};
	LONG rv;
	SCARDCONTEXT hContext;
	DWORD dwReaders;
	LPSTR mszReaders = NULL;
	char **readers = NULL;
	SCARDHANDLE hCard;
	DWORD dwActiveProtocol;
	unsigned char bRecvBuffer[MAX_BUFFER_SIZE];
	DWORD length;
	SCARD_IO_REQUEST pioRecvPci;
 	SCARD_IO_REQUEST pioSendPci;

	rv = SCardEstablishContext(SCARD_SCOPE_SYSTEM, NULL, NULL, &hContext);
	if (rv != SCARD_S_SUCCESS)
	{
		printf("SCardEstablishContext: Cannot Connect to Resource Manager %"LF"X\n", rv);
		return 1;
	}

	/* Retrieve the available readers list */
	rv = SCardListReaders(hContext, NULL, NULL, &dwReaders);
	PCSC_ERROR_EXIT(rv, "SCardListReader");

	if (dwReaders < 4)
	{
		printf("No reader found!\n");
		return -1;
	}

	mszReaders = malloc(sizeof(char)*dwReaders);
	if (mszReaders == NULL)
	{
		printf("malloc: not enough memory\n");
		goto end;
	}

	rv = SCardListReaders(hContext, NULL, mszReaders, &dwReaders);
	PCSC_ERROR_EXIT(rv, "SCardListReader");

	/* connect to the first reader */
	dwActiveProtocol = -1;
	rv = SCardConnect(hContext, mszReaders, SCARD_SHARE_EXCLUSIVE,
		SCARD_PROTOCOL_T0 | SCARD_PROTOCOL_T1, &hCard, &dwActiveProtocol);
	PCSC_ERROR_EXIT(rv, "SCardConnect")

    switch(dwActiveProtocol)
    {
        case SCARD_PROTOCOL_T0:
            pioSendPci = *SCARD_PCI_T0;
            break;
        case SCARD_PROTOCOL_T1:
            pioSendPci = *SCARD_PCI_T1;
            break;
        default:
            printf("Unknown protocol\n");
            return -1;
    }

	/* APDU select applet */
	length = sizeof(bRecvBuffer);
	rv = SCardTransmit(hCard, &pioSendPci, cmd1, sizeof cmd1,
		&pioRecvPci, bRecvBuffer, &length);
	PCSC_ERROR_EXIT(rv, "SCardTransmit")
	if ((length != 2) || (bRecvBuffer[0] != 0x90) || (bRecvBuffer[1] != 0x00))
	{
		printf("cmd1 failed (%"LF"d): %02X%02X\n", length, bRecvBuffer[length-2],
			bRecvBuffer[length-1]);
		goto end;
	}

	/* non ISO APDU */
	length = sizeof(bRecvBuffer);
	rv = SCardTransmit(hCard, &pioSendPci, cmd2, sizeof cmd2,
		&pioRecvPci, bRecvBuffer, &length);
	PCSC_ERROR_EXIT(rv, "SCardTransmit")
	if ((length != 6) || (bRecvBuffer[4] != 0x90) || (bRecvBuffer[5] != 0x00))
	{
		printf("cmd2 failed (%"LF"d) : %02X%02X\n", length,
			bRecvBuffer[length-2], bRecvBuffer[length-1]);
		goto end;
	}

	/* get the argument for cmd3 from result of cmd2 */
	memcpy(cmd3+5, bRecvBuffer, 4);

	/* non ISO APDU */
	length = sizeof(bRecvBuffer);
	rv = SCardTransmit(hCard, &pioSendPci, cmd3, sizeof cmd3,
		&pioRecvPci, bRecvBuffer, &length);
	PCSC_ERROR_EXIT(rv, "SCardTransmit")
	if ((length != 2) || (bRecvBuffer[0] != 0x90) || (bRecvBuffer[1] != 0x00))
	{
		printf("cmd3 failed (%"LF"d): %02X%02X\n", length, bRecvBuffer[length-2],
			bRecvBuffer[length-1]);
		goto end;
	}

	/* non iSO APDU */
	length = sizeof(bRecvBuffer);
	rv = SCardTransmit(hCard, &pioSendPci, cmd4, sizeof cmd4,
		&pioRecvPci, bRecvBuffer, &length);
	PCSC_ERROR_EXIT(rv, "SCardTransmit")
	if ((length != 7) || (bRecvBuffer[5] != 0x90) || (bRecvBuffer[6] != 0x00))
	{
		printf("cmd4 failed (%"LF"d): %02X%02X\n", length, bRecvBuffer[length-2],
			bRecvBuffer[length-1]);
		goto end;
	}

	printf("%02X%02X%02X\n", bRecvBuffer[2], bRecvBuffer[3], bRecvBuffer[4]);

end:
	/* We try to leave things as clean as possible */
    rv = SCardReleaseContext(hContext);
    if (rv != SCARD_S_SUCCESS)
        printf("SCardReleaseContext: %s (0x%"LF"X)\n", pcsc_stringify_error(rv),
            rv);

    /* free allocated memory */
    free(mszReaders);
    free(readers);

	return 0;
} /* main */

