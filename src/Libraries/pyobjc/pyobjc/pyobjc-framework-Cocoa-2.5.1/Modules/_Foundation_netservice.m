/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <netdb.h>


static PyObject *
makeipaddr(struct sockaddr *addr, int addrlen)
{
	char buf[NI_MAXHOST];
	int error;
	PyObject* v;

	error = getnameinfo(addr, addrlen, buf, sizeof(buf), NULL, 0,
		NI_NUMERICHOST);
	if (error) {
		v = Py_BuildValue("(is)", error, gai_strerror(error));
		PyErr_SetObject(PyExc_RuntimeError, v);
		Py_DECREF(v);
		return NULL;
	}
	return PyBytes_FromString(buf);
}

static PyObject *
makesockaddr(struct sockaddr *addr, int addrlen)
{
	if (addrlen == 0) {
		/* No address -- may be recvfrom() from known socket */
		Py_INCREF(Py_None);
		return Py_None;
	}

	switch (addr->sa_family) {

	case AF_INET:
	{
		struct sockaddr_in *a;
		PyObject *addrobj = makeipaddr(addr, sizeof(*a));
		PyObject *ret = NULL;
		if (addrobj) {
			a = (struct sockaddr_in *)addr;
			ret = Py_BuildValue("Oi", addrobj, ntohs(a->sin_port));
			Py_DECREF(addrobj);
		}
		return ret;
	}

	case AF_UNIX:
	{
		struct sockaddr_un *a = (struct sockaddr_un *) addr;
		return PyBytes_FromString(a->sun_path);
	}

	case AF_INET6:
	{
		struct sockaddr_in6 *a;
		PyObject *addrobj = makeipaddr(addr, sizeof(*a));
		PyObject *ret = NULL;
		if (addrobj) {
			a = (struct sockaddr_in6 *)addr;
			ret = Py_BuildValue("Oiii",
					    addrobj,
					    ntohs(a->sin6_port),
					    a->sin6_flowinfo,
					    a->sin6_scope_id);
			Py_DECREF(addrobj);
		}
		return ret;
	}

	/* More cases here... */

	default:
		/* If we don't know the address family, don't raise an
		   exception -- return it as a tuple. */
		return Py_BuildValue("is#",
				     addr->sa_family,
				     addr->sa_data,
				     sizeof(addr->sa_data));

	}
}

static PyObject* call_NSNetService_addresses(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	PyObject* result;
	struct objc_super super;
	NSArray*  res;
	int len, i;
	NSData* item;

	if  (!PyArg_ParseTuple(arguments, "")) {
		return NULL;
	}

	PyObjC_DURING
		PyObjC_InitSuper(&super, 
			PyObjCSelector_GetClass(method),
			PyObjCObject_GetObject(self));

			
		res = ((id(*)(struct objc_super*, SEL))objc_msgSendSuper)(&super, @selector(addresses));
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		res = nil;
	PyObjC_ENDHANDLER

	if (res == nil && PyErr_Occurred()) {
		return NULL;
	}

	if (res == nil) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	len = [res count];
	result = PyTuple_New(len);
	if (result == NULL) {
		return NULL;
	}

	for (i = 0; i < len; i++) {

		PyObject* v;

		item = [res objectAtIndex:i];

		v = makesockaddr((struct sockaddr*)[item bytes], [item length]);
		if (v == NULL) {
			Py_DECREF(result);
			return NULL;
		}
		PyTuple_SetItem(result, i, v);
	}

	return result;
}


static int setup_nsnetservice(PyObject* m __attribute__((__unused__)))
{
	Class classNSNetService = objc_lookUpClass("NSNetService");
	if (classNSNetService == NULL) {
		return 0;
	}

	if (PyObjC_RegisterMethodMapping(
		classNSNetService,
		@selector(addresses),
		call_NSNetService_addresses,
		PyObjCUnsupportedMethod_IMP) < 0) {

		return -1;
	}

	return 0;
}
