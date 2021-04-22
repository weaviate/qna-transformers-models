import unittest
import requests
import time


class SmokeTest(unittest.TestCase):
    def _waitForStartup(self):
        url = 'http://localhost:8000/.well-known/ready'

        for i in range(0, 100):
            try:
                res = requests.get(url)
                if res.status_code == 204:
                    return
                else:
                    raise Exception(
                            "status code is {}".format(res.status_code))
            except Exception as e:
                print("Attempt {}: {}".format(i, e))
                time.sleep(1)

        raise Exception("did not start up")

    def testVectorizing(self):
        self._waitForStartup()
        url = 'http://localhost:8000/answers/'

        req_body = {'text': 'John is 20 years old', 'question': 'how old is john?'}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertEqual(req_body['question'], resBody['question'])
        self.assertEqual(str(resBody['answer']).strip(), "20 years old")
        self.assertGreaterEqual(resBody['certainty'], 0.1)

        req_body = {'text': 'John is 20 years old', 'question': 'this is wrong question'}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertEqual(req_body['question'], resBody['question'])
        if resBody['certainty'] != None:
            self.assertLessEqual(resBody['certainty'], 0.5)

        req_body = {'text': 'John is 20 years old', 'question': 'This question has no meaning or has it?'}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertEqual(req_body['question'], resBody['question'])
        if resBody['answer'] != None:
            self.assertEqual(len(resBody['answer']), 0)
        if resBody['certainty'] != None:
            self.assertEqual(resBody['certainty'], 0.0)

        req_body = {"question": "how old is john?", "text": "Praesent pulvinar semper feugiat. Sed eros nisl, volutpat ut dolor et, consectetur finibus libero. Aenean molestie, neque vel aliquet ultrices, nisl magna molestie justo, id fringilla neque urna a orci. Curabitur a leo sed enim blandit bibendum. Sed a risus in tortor varius tristique ut ac purus. Proin vitae bibendum magna. Donec sit amet fermentum arcu. Nulla pharetra hendrerit elementum. In varius pretium leo, a auctor tortor. Phasellus rutrum lacus quis imperdiet sagittis. Proin et scelerisque eros. Suspendisse convallis at erat et condimentum... Donec eu orci eu nibh ullamcorper varius a ut quam. Mauris tempus semper tincidunt. Aliquam eu justo vestibulum, semper sapien ut, sollicitudin justo. Suspendisse potenti. Proin ultricies feugiat tortor non viverra. Aliquam eleifend mollis orci ut lacinia. Etiam tincidunt sem velit, vel consequat arcu finibus nec. Donec tincidunt sem quam, eu lacinia orci blandit at. Donec fermentum, lacus eu congue viverra, nisl risus hendrerit ex, et feugiat metus nisi nec dui. In in elit a nunc elementum ullamcorper. In non nunc in dolor placerat malesuada vitae sed nulla. In vitae metus sed mi laoreet ultricies eget in quam. Quisque consectetur ipsum in lorem congue porta. Maecenas scelerisque, mauris ac molestie malesuada, eros orci blandit quam, vitae vulputate lectus eros sit amet sem. Duis luctus venenatis risus ut lacinia. Aenean sed enim volutpat, elementum elit eget, semper augue. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean feugiat tellus odio, aliquam dignissim odio placerat vel. Curabitur id dolor sed mi scelerisque dictum. Sed tempus varius dolor id condimentum. Etiam rutrum vestibulum odio. Ut a convallis arcu, eu porta magna. Fusce eu euismod justo, at malesuada quam. Duis dignissim id ipsum a interdum. Proin et urna faucibus tellus accumsan bibendum id sed erat. Ut at eros ac nibh faucibus sollicitudin. Ut scelerisque arcu libero, eu finibus felis porta in. Phasellus varius gravida massa. Nulla tortor augue, eleifend et orci non, venenatis scelerisque ex. Sed quis diam eu lorem pretium lobortis. Morbi malesuada, leo id egestas ultrices, magna justo euismod enim, et lacinia magna nulla eu ex. There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc. John is 20 years old. Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of \"de Finibus Bonorum et Malorum\" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, \"Lorem ipsum dolor sit amet..\", comes from a line in section 1.10.32. The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from \"de Finibus Bonorum et Malorum\" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham. Duis euismod odio a dolor porttitor, sit amet imperdiet enim lacinia. Vestibulum mattis gravida metus, at vulputate metus consequat eleifend. Nunc quis risus leo. Nullam at augue eget odio tincidunt facilisis sed in sem. Pellentesque consectetur, ex quis maximus sagittis, felis tortor suscipit tortor, a sagittis nisl arcu quis sem. Sed finibus, eros quis suscipit elementum, est sapien gravida purus, non ullamcorper mauris arcu quis nulla. Suspendisse at felis vitae neque finibus lobortis sed eu urna. Ut suscipit laoreet erat, vitae hendrerit lorem gravida vel. Proin quis risus nec tortor facilisis mollis eu nec urna. In at ullamcorper purus. Aenean ac ligula orci. Mauris porta fermentum ante, ut tempor nisi dignissim ac. Vivamus pulvinar a velit et interdum. Donec dolor dui, pellentesque quis vestibulum egestas, dapibus non erat. Praesent vitae euismod turpis. Proin efficitur rutrum ante, at posuere enim ultricies ac. Pellentesque sed finibus lorem. Morbi dapibus posuere lacus, vitae luctus libero suscipit ut. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Fusce lacus diam, bibendum eu suscipit quis, hendrerit sit amet quam. Nam eget sodales leo. Vivamus varius nec purus ut facilisis. In erat diam, rutrum et quam id, vestibulum sollicitudin ligula. Sed luctus porttitor diam elementum faucibus. Phasellus pellentesque purus vitae nulla sodales, a efficitur felis fermentum. Mauris vulputate dapibus metus commodo maximus. Nulla eget sapien eu nibh efficitur porttitor nec nec massa. Nullam rutrum lorem ut justo viverra blandit. Praesent a erat lorem. Ut eget quam id ipsum venenatis condimentum auctor et turpis. Nullam et bibendum ipsum. Duis varius et nisi eget maximus. Duis commodo feugiat porttitor. Sed sed rutrum erat. Morbi auctor urna a porta suscipit. Maecenas eget maximus nisi, at euismod eros."}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertEqual(req_body['question'], resBody['question'])
        self.assertGreaterEqual(len(resBody['answer']), 1)
        self.assertGreaterEqual(resBody['certainty'], 0.1)

        req_body = {'text': '', 'question': ''}
        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        self.assertEqual(req_body['text'], resBody['text'])
        self.assertEqual(req_body['question'], resBody['question'])
        self.assertIsNone(resBody['answer'])
        self.assertIsNone(resBody['certainty'])


if __name__ == "__main__":
    unittest.main()
