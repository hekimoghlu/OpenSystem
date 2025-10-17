/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#include "rpcapd.h"
#include <pcap.h>		// for PCAP_ERRBUF_SIZE
#include "fmtutils.h"
#include "portability.h"
#include "fileconf.h"
#include "log.h"

#include "win32-svc.h"	// for Win32 service stuff

static SERVICE_STATUS_HANDLE service_status_handle;
static SERVICE_STATUS service_status;

static void WINAPI svc_main(DWORD argc, char **argv);
static void WINAPI svc_control_handler(DWORD Opcode);
static void update_svc_status(DWORD state, DWORD progress_indicator);

BOOL svc_start(void)
{
	BOOL rc;
	SERVICE_TABLE_ENTRY ste[] =
	{
		{ PROGRAM_NAME, svc_main },
		{ NULL, NULL }
	};
	char string[PCAP_ERRBUF_SIZE];

	// This call is blocking. A new thread is created which will launch
	// the svc_main() function
	if ((rc = StartServiceCtrlDispatcher(ste)) == 0) {
		pcap_fmt_errmsg_for_win32_err(string, sizeof (string),
		    GetLastError(), "StartServiceCtrlDispatcher() failed");
		rpcapd_log(LOGPRIO_ERROR, "%s", string);
	}

	return rc; // FALSE if this is not started as a service
}

static void WINAPI
svc_control_handler(DWORD Opcode)
{
	switch(Opcode)
	{
		case SERVICE_CONTROL_STOP:
			//
			// XXX - is this sufficient to clean up the service?
			// To be really honest, only the main socket and
			// such these stuffs are cleared; however the threads
			// that are running are not stopped.
			// This can be seen by placing a breakpoint at the
			// end of svc_main(), in which you will see that is
			// never reached. However, as soon as you set the
			// service status to "stopped",	the
			// StartServiceCtrlDispatcher() returns and the main
			// thread ends. Then, Win32 has a good automatic
			// cleanup, so that all the threads which are still
			// running are stopped when the main thread ends.
			//
			send_shutdown_notification();

			update_svc_status(SERVICE_STOP_PENDING, 0);
			break;

		/*
			Pause and Continue have an usual meaning and they are used just to be able
			to change the running parameters at run-time. In other words, they act
			like the SIGHUP signal on UNIX. All the running threads continue to run and
			they are not paused at all.
			Particularly,
			- PAUSE does nothing
			- CONTINUE re-reads the configuration file and creates the new threads that
			can be needed according to the new configuration.
		*/
		case SERVICE_CONTROL_PAUSE:
			update_svc_status(SERVICE_PAUSED, 0);
			break;

		case SERVICE_CONTROL_CONTINUE:
			update_svc_status(SERVICE_RUNNING, 0);
			//
			// Tell the main loop to re-read the configuration.
			//
			send_reread_configuration_notification();
			break;

		case SERVICE_CONTROL_INTERROGATE:
			// Fall through to send current status.
			//	WARNING: not implemented
			update_svc_status(SERVICE_RUNNING, 0);
			MessageBox(NULL, "Not implemented", "warning", MB_OK);
			break;

		case SERVICE_CONTROL_PARAMCHANGE:
			//
			// Tell the main loop to re-read the configuration.
			//
			send_reread_configuration_notification();
			break;
	}

	// Send current status.
	return;
}

static void WINAPI
svc_main(DWORD argc, char **argv)
{
	service_status_handle = RegisterServiceCtrlHandler(PROGRAM_NAME, svc_control_handler);

	if (!service_status_handle)
		return;

	service_status.dwServiceType = SERVICE_WIN32_OWN_PROCESS | SERVICE_INTERACTIVE_PROCESS;
	service_status.dwControlsAccepted = SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_PAUSE_CONTINUE | SERVICE_ACCEPT_PARAMCHANGE;
	// | SERVICE_ACCEPT_SHUTDOWN ;
	update_svc_status(SERVICE_RUNNING, 0);

	//
	// Service requests until we're told to stop.
	//
	main_startup();

	//
	// It returned, so we were told to stop.
	//
	update_svc_status(SERVICE_STOPPED, 0);
}

static void
update_svc_status(DWORD state, DWORD progress_indicator)
{
	service_status.dwWin32ExitCode = NO_ERROR;
	service_status.dwCurrentState = state;
	service_status.dwCheckPoint = progress_indicator;
	service_status.dwWaitHint = 0;
	SetServiceStatus(service_status_handle, &service_status);
}

/*
sc create rpcapd DisplayName= "Remote Packet Capture Protocol v.0 (experimental)" binpath= "C:\cvsroot\winpcap\wpcap\PRJ\Debug\rpcapd -d -f rpcapd.ini"
sc description rpcapd "Allows to capture traffic on this host from a remote machine."
*/
