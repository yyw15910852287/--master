import allure
import pytest
import yaml
from pytest_assume.plugin import assume

# @pytest.fixture(scope="module")
# def open():
#     print("open")
#     yield
#     print("close")
#
# class Test1():
#
#     @pytest.mark.parametrize("a,b",[(1,2),(3,4)])
#     def test_a(self,a,b):
#         print("testa")
#         # assert 1 == 1
#         print(f"a:{a}")
#         print(f"b:{b}")
#         assume(1 == 1)
#
#     def test_b(self):
#         print("testb")
#         # assert 'i' in 'inside'
#         assume('i' in 'inside')
#
#     def test_f(self):
#         print("testf")
#         # assert 'i' in 'inside'
#         assume('i' in 'inside')
#
# class Test2():
#     def test_c(self):
#         print("testc")
#         # assert 1 == 2
#         assume(1 == 1)
#
#     def test_d(self):
#         print("testd")
#         # assert 'i' in 'inside'
#         assume('i' in 'inside')

class Test3():
    # test_data = ["admin1", "admin2"]
    #
    # @pytest.fixture(scope="function")
    # def login(self,request):
    #     user = request.param
    #     print("登录账户：%s" % user)
    #     return user
    #
    # @pytest.mark.parametrize("login", test_data, indirect=True)
    # def test_login(self,login):
    #     """登录用例"""
    #     a = login
    #     print("测试用例中login的返回值:%s" % a)
    #     assert a != ""

    @pytest.fixture(scope='function')
    def login(self,request):
        '''登录函数'''
        user = request.param['user']
        password = request.param['password']
        print('用户名：%s' % user)
        print('密码：%s' % password)
        return 'hello'

    login_data = [{'user': 'admin', 'password': '123456'}, {'user': 'admin', 'password': '1'}]

    # 装饰器
    @pytest.mark.parametrize('login',  login_data, indirect=True)
    def test_login(self,login):
        '''登录测试用例'''
        result = login
        assert result == 'hello'
@allure.feature("测试登录功能")
class Test4():
    @pytest.mark.parametrize("env",yaml.safe_load(open("test_yaml.yml")))
    def test_11(self,env):
        print(yaml.safe_load(open("test_yaml.yml")))
        print(env)
        with allure.step("关键步骤2"):
            if "test" == env:
                print("yes")


