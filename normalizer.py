import xml.etree.ElementTree as Et

tree = Et.parse("corpus/H9E.xml")
root = tree.getroot().iterfind("s")

print(next())