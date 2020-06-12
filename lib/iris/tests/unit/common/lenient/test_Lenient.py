# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.lenient.Lenient`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from collections import Iterable

from iris.common.lenient import LENIENT_PROTECTED, Lenient, qualname


class Test___init__(tests.IrisTest):
    def test_default(self):
        lenient = Lenient()
        expected = dict(active=None)
        self.assertEqual(expected, lenient.__dict__)

    def test_args_service_str(self):
        service = "service1"
        lenient = Lenient(service)
        expected = dict(active=None, service1=True)
        self.assertEqual(expected, lenient.__dict__)

    def test_args_services_str(self):
        services = ("service1", "service2")
        lenient = Lenient(*services)
        expected = dict(active=None, service1=True, service2=True)
        self.assertEqual(expected, lenient.__dict__)

    def test_args_services_callable(self):
        def service1():
            pass

        def service2():
            pass

        services = (service1, service2)
        lenient = Lenient(*services)
        expected = {
            "active": None,
            qualname(service1): True,
            qualname(service2): True,
        }
        self.assertEqual(expected, lenient.__dict__)

    def test_kwargs_client_str(self):
        client = dict(client1="service1")
        lenient = Lenient(**client)
        expected = dict(active=None, client1=("service1",))
        self.assertEqual(expected, lenient.__dict__)

    def test_kwargs_clients_str(self):
        clients = dict(client1="service1", client2="service2")
        lenient = Lenient(**clients)
        expected = dict(
            active=None, client1=("service1",), client2=("service2",)
        )
        self.assertEqual(expected, lenient.__dict__)

    def test_kwargs_clients_callable(self):
        def client1():
            pass

        def client2():
            pass

        def service1():
            pass

        def service2():
            pass

        qualname_client1 = qualname(client1)
        qualname_client2 = qualname(client2)
        clients = {
            qualname_client1: service1,
            qualname_client2: (service1, service2),
        }
        lenient = Lenient(**clients)
        expected = {
            "active": None,
            qualname(client1): (qualname(service1),),
            qualname(client2): (qualname(service1), qualname(service2)),
        }
        self.assertEqual(expected, lenient.__dict__)


class Test___call__(tests.IrisTest):
    def setUp(self):
        self.client = "myclient"
        self.lenient = Lenient()

    def test_missing_service_str(self):
        self.assertFalse(self.lenient("myservice"))

    def test_missing_service_callable(self):
        def myservice():
            pass

        self.assertFalse(self.lenient(myservice))

    def test_disabled_service_str(self):
        service = "myservice"
        self.lenient.__dict__[service] = False
        self.assertFalse(self.lenient(service))

    def test_disable_service_callable(self):
        def myservice():
            pass

        qualname_service = qualname(myservice)
        self.lenient.__dict__[qualname_service] = False
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_no_active_client(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_no_active_client(self):
        def myservice():
            pass

        qualname_service = qualname(myservice)
        self.lenient.__dict__[qualname_service] = True
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_no_registered_services(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_active_client_with_no_registered_services(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = qualname(myservice)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname(myclient)
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_unmatched_registered_services(
        self,
    ):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = ("service1", "service2")
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_active_client_with_unmatched_registered_services(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = qualname(myservice)
        qualname_client = qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = ("service1", "service2")
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_registered_services(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = ("service1", "service2", service)
        self.assertTrue(self.lenient(service))

    def test_service_callable_with_active_client_with_registered_services(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = qualname(myservice)
        qualname_client = qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = (
            "service1",
            "service2",
            qualname_service,
        )
        self.assertTrue(self.lenient(myservice))

    def test_service_str_with_active_client_with_unmatched_registered_service_str(
        self,
    ):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = "serviceXXX"
        self.assertFalse(self.lenient(service))

    def test_service_callable_with_active_client_with_unmatched_registered_service_str(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = qualname(myservice)
        qualname_client = qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = f"{qualname_service}XXX"
        self.assertFalse(self.lenient(myservice))

    def test_service_str_with_active_client_with_registered_service_str(self):
        service = "myservice"
        self.lenient.__dict__[service] = True
        self.lenient.__dict__["active"] = self.client
        self.lenient.__dict__[self.client] = service
        self.assertTrue(self.lenient(service))

    def test_service_callable_with_active_client_with_registered_service_str(
        self,
    ):
        def myservice():
            pass

        def myclient():
            pass

        qualname_service = qualname(myservice)
        qualname_client = qualname(myclient)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.__dict__["active"] = qualname_client
        self.lenient.__dict__[qualname_client] = qualname_service
        self.assertTrue(self.lenient(myservice))


class Test___contains__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_in(self):
        self.assertIn("active", self.lenient)

    def test_not_in(self):
        self.assertNotIn("ACTIVATE", self.lenient)


class Test___getattr__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_in(self):
        self.assertIsNone(self.lenient.active)

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(AttributeError, emsg):
            _ = self.lenient.wibble


class Test__getitem__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_in(self):
        self.assertIsNone(self.lenient["active"])

    def test_in_callable(self):
        def service():
            pass

        qualname_service = qualname(service)
        self.lenient.__dict__[qualname_service] = True
        self.assertTrue(self.lenient[service])

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(KeyError, emsg):
            _ = self.lenient["wibble"]

    def test_not_in_callable(self):
        def service():
            pass

        qualname_service = qualname(service)
        emsg = f"Invalid .* option, got '{qualname_service}'."
        with self.assertRaisesRegex(KeyError, emsg):
            _ = self.lenient[service]


class Test___setattr__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(AttributeError, emsg):
            self.lenient.wibble = None

    def test_in_value_str(self):
        client = "client"
        service = "service"
        self.lenient.__dict__[client] = None
        self.lenient.client = service
        self.assertEqual(self.lenient.__dict__[client], (service,))

    def test_in_value_callable(self):
        def service():
            pass

        client = "client"
        qualname_service = qualname(service)
        self.lenient.__dict__[client] = None
        self.lenient.client = service
        self.assertEqual(self.lenient.__dict__[client], (qualname_service,))

    def test_in_value_bool(self):
        client = "client"
        self.lenient.__dict__[client] = None
        self.lenient.client = True
        self.assertTrue(self.lenient.__dict__[client])
        self.assertFalse(isinstance(self.lenient.__dict__[client], Iterable))

    def test_in_value_iterable(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.__dict__[client] = None
        self.lenient.client = services
        self.assertEqual(self.lenient.__dict__[client], services)

    def test_in_value_iterable_callable(self):
        def service1():
            pass

        def service2():
            pass

        client = "client"
        self.lenient.__dict__[client] = None
        qualname_services = (qualname(service1), qualname(service2))
        self.lenient.client = (service1, service2)
        self.assertEqual(self.lenient.__dict__[client], qualname_services)

    def test_active_iterable(self):
        self.assertIsNone(self.lenient.__dict__["active"])
        emsg = "Invalid .* option 'active'"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.active = (None,)

    def test_active_str(self):
        active = "active"
        client = "client1"
        self.assertIsNone(self.lenient.__dict__[active])
        self.lenient.active = client
        self.assertEqual(self.lenient.__dict__[active], client)

    def test_active_callable(self):
        def client():
            pass

        active = "active"
        qualname_client = qualname(client)
        self.assertIsNone(self.lenient.__dict__[active])
        self.lenient.active = client
        self.assertEqual(self.lenient.__dict__[active], qualname_client)


class Test___setitem__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_not_in(self):
        emsg = "Invalid .* option, got 'wibble'."
        with self.assertRaisesRegex(KeyError, emsg):
            self.lenient["wibble"] = None

    def test_in_value_str(self):
        client = "client"
        service = "service"
        self.lenient.__dict__[client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[client], (service,))

    def test_callable_in_value_str(self):
        def client():
            pass

        service = "service"
        qualname_client = qualname(client)
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[qualname_client], (service,))

    def test_in_value_callable(self):
        def service():
            pass

        client = "client"
        qualname_service = qualname(service)
        self.lenient.__dict__[client] = None
        self.lenient[client] = service
        self.assertEqual(self.lenient.__dict__[client], (qualname_service,))

    def test_callable_in_value_callable(self):
        def client():
            pass

        def service():
            pass

        qualname_client = qualname(client)
        qualname_service = qualname(service)
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = service
        self.assertEqual(
            self.lenient.__dict__[qualname_client], (qualname_service,)
        )

    def test_in_value_bool(self):
        client = "client"
        self.lenient.__dict__[client] = None
        self.lenient[client] = True
        self.assertTrue(self.lenient.__dict__[client])
        self.assertFalse(isinstance(self.lenient.__dict__[client], Iterable))

    def test_callable_in_value_bool(self):
        def client():
            pass

        qualname_client = qualname(client)
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = True
        self.assertTrue(self.lenient.__dict__[qualname_client])
        self.assertFalse(
            isinstance(self.lenient.__dict__[qualname_client], Iterable)
        )

    def test_in_value_iterable(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.__dict__[client] = None
        self.lenient[client] = services
        self.assertEqual(self.lenient.__dict__[client], services)

    def test_callable_in_value_iterable(self):
        def client():
            pass

        qualname_client = qualname(client)
        services = ("service1", "service2")
        self.lenient.__dict__[qualname_client] = None
        self.lenient[client] = services
        self.assertEqual(self.lenient.__dict__[qualname_client], services)

    def test_in_value_iterable_callable(self):
        def service1():
            pass

        def service2():
            pass

        client = "client"
        self.lenient.__dict__[client] = None
        qualname_services = (qualname(service1), qualname(service2))
        self.lenient[client] = (service1, service2)
        self.assertEqual(self.lenient.__dict__[client], qualname_services)

    def test_callable_in_value_iterable_callable(self):
        def client():
            pass

        def service1():
            pass

        def service2():
            pass

        qualname_client = qualname(client)
        self.lenient.__dict__[qualname_client] = None
        qualname_services = (qualname(service1), qualname(service2))
        self.lenient[client] = (service1, service2)
        self.assertEqual(
            self.lenient.__dict__[qualname_client], qualname_services
        )

    def test_active_iterable(self):
        active = "active"
        self.assertIsNone(self.lenient.__dict__[active])
        emsg = "Invalid .* option 'active'"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient[active] = (None,)

    def test_active_str(self):
        active = "active"
        client = "client1"
        self.assertIsNone(self.lenient.__dict__[active])
        self.lenient[active] = client
        self.assertEqual(self.lenient.__dict__[active], client)

    def test_active_callable(self):
        def client():
            pass

        active = "active"
        qualname_client = qualname(client)
        self.assertIsNone(self.lenient.__dict__[active])
        self.lenient[active] = client
        self.assertEqual(self.lenient.__dict__[active], qualname_client)


class Test_context(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()
        self.default = dict(active=None)

    def copy(self):
        return self.lenient.__dict__.copy()

    def test_nop(self):
        pre = self.copy()
        with self.lenient.context():
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        self.assertEqual(context, self.default)
        self.assertEqual(post, self.default)

    def test_active_str(self):
        client = "client"
        pre = self.copy()
        with self.lenient.context(active=client):
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        self.assertEqual(context, dict(active=client))
        self.assertEqual(post, self.default)

    def test_active_callable(self):
        def client():
            pass

        pre = self.copy()
        with self.lenient.context(active=client):
            context = self.copy()
        post = self.copy()
        qualname_client = qualname(client)
        self.assertEqual(pre, self.default)
        self.assertEqual(context, dict(active=qualname_client))
        self.assertEqual(post, self.default)

    def test_kwargs(self):
        client = "client"
        self.lenient.__dict__["service1"] = False
        self.lenient.__dict__["service2"] = False
        pre = self.copy()
        with self.lenient.context(active=client, service1=True, service2=True):
            context = self.copy()
        post = self.copy()
        default = dict(active=None, service1=False, service2=False)
        self.assertEqual(pre, default)
        expected = dict(active=client, service1=True, service2=True)
        self.assertEqual(context, expected)
        self.assertEqual(post, default)

    def test_args_str(self):
        client = "client"
        services = ("service1", "service2")
        pre = self.copy()
        with self.lenient.context(*services, active=client):
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        expected = dict(active=client, client=services)
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)

    def test_args_callable(self):
        def service1():
            pass

        def service2():
            pass

        client = "client"
        services = (service1, service2)
        pre = self.copy()
        with self.lenient.context(*services, active=client):
            context = self.copy()
        post = self.copy()
        qualname_services = tuple([qualname(service) for service in services])
        self.assertEqual(pre, self.default)
        expected = dict(active=client, client=qualname_services)
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)

    def test_context_runtime(self):
        services = ("service1", "service2")
        pre = self.copy()
        with self.lenient.context(*services):
            context = self.copy()
        post = self.copy()
        self.assertEqual(pre, self.default)
        expected = dict(active="context", context=services)
        self.assertEqual(context, expected)
        self.assertEqual(post, self.default)


class Test_register_client(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_not_protected(self):
        emsg = "Cannot register .* protected non-client"
        for protected in LENIENT_PROTECTED:
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.register_client(protected, "service")

    def test_str_service_str(self):
        client = "client"
        services = "service"
        self.lenient.register_client(client, services)
        self.assertIn(client, self.lenient.__dict__)
        self.assertEqual(self.lenient.__dict__[client], (services,))

    def test_str_services_str(self):
        client = "client"
        services = ("service1", "service2")
        self.lenient.register_client(client, services)
        self.assertIn(client, self.lenient.__dict__)
        self.assertEqual(self.lenient.__dict__[client], services)

    def test_callable_service_callable(self):
        def client():
            pass

        def service():
            pass

        qualname_client = qualname(client)
        qualname_service = qualname(service)
        self.lenient.register_client(client, service)
        self.assertIn(qualname_client, self.lenient.__dict__)
        self.assertEqual(
            self.lenient.__dict__[qualname_client], (qualname_service,)
        )

    def test_callable_services_callable(self):
        def client():
            pass

        def service1():
            pass

        def service2():
            pass

        qualname_client = qualname(client)
        qualname_services = (qualname(service1), qualname(service2))
        self.lenient.register_client(client, (service1, service2))
        self.assertIn(qualname_client, self.lenient.__dict__)
        self.assertEqual(
            self.lenient.__dict__[qualname_client], qualname_services
        )

    def test_services_empty(self):
        emsg = "Require at least one .* lenient client service."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.register_client("client", ())


class Test_register_service(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_str(self):
        service = "service"
        self.assertNotIn(service, self.lenient.__dict__)
        self.lenient.register_service(service)
        self.assertIn(service, self.lenient.__dict__)
        self.assertFalse(isinstance(self.lenient.__dict__[service], Iterable))
        self.assertTrue(self.lenient.__dict__[service])

    def test_callable(self):
        def service():
            pass

        qualname_service = qualname(service)
        self.assertNotIn(qualname_service, self.lenient.__dict__)
        self.lenient.register_service(service)
        self.assertIn(qualname_service, self.lenient.__dict__)
        self.assertFalse(
            isinstance(self.lenient.__dict__[qualname_service], Iterable)
        )
        self.assertTrue(self.lenient.__dict__[qualname_service])

    def test_not_protected(self):
        emsg = "Cannot register .* protected non-service"
        for protected in LENIENT_PROTECTED:
            self.lenient.__dict__[protected] = None
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.register_service("active")


class Test_unregister_client(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_not_protected(self):
        emsg = "Cannot unregister .* protected non-client"
        for protected in LENIENT_PROTECTED:
            self.lenient.__dict__[protected] = None
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.unregister_client(protected)

    def test_not_in(self):
        emsg = "Cannot unregister unknown .* client"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_client("client")

    def test_not_client(self):
        client = "client"
        self.lenient.__dict__[client] = True
        emsg = "Cannot unregister .* non-client"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_client(client)

    def test_not_client_callable(self):
        def client():
            pass

        qualname_client = qualname(client)
        self.lenient.__dict__[qualname_client] = True
        emsg = "Cannot unregister .* non-client"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_client(client)

    def test_str(self):
        client = "client"
        self.lenient.__dict__[client] = (None,)
        self.lenient.unregister_client(client)
        self.assertNotIn(client, self.lenient.__dict__)

    def test_callable(self):
        def client():
            pass

        qualname_client = qualname(client)
        self.lenient.__dict__[qualname_client] = (None,)
        self.lenient.unregister_client(client)
        self.assertNotIn(qualname_client, self.lenient.__dict__)


class Test_unregister_service(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_not_protected(self):
        emsg = "Cannot unregister .* protected non-service"
        for protected in LENIENT_PROTECTED:
            self.lenient.__dict__[protected] = None
            with self.assertRaisesRegex(ValueError, emsg):
                self.lenient.unregister_service(protected)

    def test_not_in(self):
        emsg = "Cannot unregister unknown .* service"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_service("service")

    def test_not_service(self):
        service = "service"
        self.lenient.__dict__[service] = (None,)
        emsg = "Cannot unregister .* non-service"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_service(service)

    def test_not_service_callable(self):
        def service():
            pass

        qualname_service = qualname(service)
        self.lenient.__dict__[qualname_service] = (None,)
        emsg = "Cannot unregister .* non-service"
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient.unregister_service(service)

    def test_str(self):
        service = "service"
        self.lenient.__dict__[service] = True
        self.lenient.unregister_service(service)
        self.assertNotIn(service, self.lenient.__dict__)

    def test_callable(self):
        def service():
            pass

        qualname_service = qualname(service)
        self.lenient.__dict__[qualname_service] = True
        self.lenient.unregister_service(service)
        self.assertNotIn(qualname_service, self.lenient.__dict__)


if __name__ == "__main__":
    tests.main()