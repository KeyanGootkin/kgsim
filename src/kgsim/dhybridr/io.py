"""Handle the input and out files for dHybridR simulations"""
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Imports                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from datetime import datetime
from importlib.metadata import version

from kgsim.templates import dHybridRtemplate

from kbasic.parsing import File
from numpy import float32, float64, ndarray, unique, r_

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Errors                                      <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class dHybridRInputError(Exception):
    """Exceptions relating to dHybridR's input file"""

class EmptyDHRInputSection(dHybridRInputError):
    """given an empty section of a dHybridR input file"""

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                   Functions                                    <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# fortran parsing
def input_match(input_code: str):
    """
    matches objects for dHybridR input files to equivilent python objects
    :param input_code:
    :return: python_code
    """
    match input_code:
        ###### Booleans ######
        case '.true.':
            return True
        case '.false.':
            return False
        ###### numbers ######
        case number if number.isnumeric() or (number[0]=='-' and number[1:].isnumeric()):
            # int
            return int(number)
        case number if all([
            x.isnumeric() if len(x) > 0 else True for x in number.split('.')
        ]) or (number[0]=='-' and all([
            x.isnumeric() if len(x) > 0 else True for x in number[1:].split('.')
        ])):
            # float
            return float(number)
        case number if all([
            "d" in number,
            "." in number,
            "/" not in number
        ]):
            # scientific notation
            v, exp = number.split("d")
            return float(v) * 10. ** int(exp)
        ###### strings ######
        case string if all([
            string[0] == '"',
            string[-1] == '"',
            "," not in string
        ]):
            return str(string[1:-1])
        ###### else ######
        case _:
            raise dHybridRInputError(f"I couldn't recognize the code you gave me\n{input_code}")

def python_match_input(python_code) -> str:
    """
    matches python objects to dHybridR input file equivilents
    :param python_code: python object
    :return: input equivilent
    """
    match python_code:
        case True:
            return '.true.'
        case False:
            return '.false.'
        case number if type(number) == int:
            return f"{python_code}"
        case number if type(number) in [float, float64, float32]:
            if str(number).rsplit('.', maxsplit=1)[-1] == '0':
                return f"{int(python_code)}."
            elif 'e' in str(number):
                return str(number).replace('e', 'd')
            return f"{python_code}"
        case string if type(string) == str:
            return f'"{python_code}"'

def python2input(code):
    """
    take an iterable and convert each object to dHybridR input file equivalents
    :param code: iterable containing python objects
    :return: list of input file equivalents
    """
    if not type(code) in [tuple, list, ndarray]:
        return python_match_input(code)
    input_code = [f"{python_match_input(x)}," for x in code]
    return "".join(input_code)

def input2python(code):
    """
    converts dHybridR input file code into equivilent python code
    :param code:
    :return:
    """
    python_code = [input_match(x.strip(" ")) for x in code.split(',') if len(x.strip(" ")) != 0]
    if len(python_code) == 1:
        return python_code[0]
    return python_code

def is_input_header_boarder(line: str) -> bool:
    """docstring"""
    return len(chars:=unique(list(line)))==3 and all(chars == unique(list("! -")))

def is_input_section_header(line: str) -> bool:
    """docstring"""
    return (line.startswith("!---") or line.startswith("! ---")) 

def is_input_species_section_header(line: str, species: int = None) -> bool:
    """docstring"""
    if species:
        return is_input_section_header(line) and f"species {species})" in line.lower()
    return any(
        is_input_section_header(line) and f"species {sp}" in line.lower() for sp in range(10)
        )

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Classes                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class InputParameter:
    """a class to format input parameters for dHybridR input files"""
    def __init__(self, name: str, value, comment: str = None) -> None:
        self.name = name
        self.input_name = name if type(value) not in [list, tuple, ndarray] \
                            else f"{name}(1:{len(value)})"
        self.value = value
        self.comment = "" if comment is None else "!"+comment

    def __str__(self) -> str:
        # two tabs ->> parameter=value -> make it 40 characters -> add comment
        show_string = '\t'*2+f"{self.input_name}={python2input(self.value)}".ljust(40)+self.comment
        return show_string if self.value is not None else f"!{show_string}"

    def __repr__(self) -> str:
        return f"{self.name}: {self.value}"


class InputSection:
    """a class to read sections of input files for dHybridR simulations"""
    def __init__(self, lines: list) -> None:
        self.lines = lines
        self.section_header: str = self.lines[0]
        #check if its an empty section
        while len(self.lines)>1 and self.lines[1].startswith("!"): del self.lines[1]
        if len(self.lines)==1: raise EmptyDHRInputSection(lines)
        #else parse the name and make sure its bound correctly
        self.section_name: str = self.lines[1].strip()
        self.params = {}
        for line in self.lines[3:-1]:
            #check for comments
            if line.strip().startswith("!"): continue
            #otherwise its code
            [code, *comment] = line.split("!") if "!" in line else [line.strip(), None]
            #pls don't put multiple !s on a code line :(
            comment = comment[0]
            name, value = code.strip().split("=")
            #if its a list itll have the dumb lil (1:lenth) thing
            if "(" in name: name = name.split("(")[0]
            new_param = InputParameter(
                name, input2python(value),
                comment = None if comment is None else comment.strip()
            )
            self.params[name] = new_param
            setattr(self, name, new_param)

    def __repr__(self) -> str:
        return "\n".join(self.lines)

    def __str__(self) -> str:
        return "\n".join([
            self.section_header,
            self.section_name,
            "{",
            "\n".join([str(p) for p in self.params.values()]),
            "}"
        ])


class SpeciesInput:
    """a class to read species sections of input files for dHybridR simulations"""
    def __init__(self, lines: list, i: int) -> None:
        self.lines = lines
        self.number = i
        self.sections = {}
        i = 0
        while i+1<len(self.lines):
            line = self.lines[i]
            next_line = self.lines[i+1]
            if is_input_section_header(line) and not is_input_section_header(next_line):
                sec_start = i
                while not i+1==len(self.lines) and not is_input_section_header(
                    self.lines[i+1]
                    ): i+=1
                sec_end = i+1
                if sec_end-sec_start>2:
                    new_sec = InputSection(self.lines[sec_start:sec_end])
                    self.sections[new_sec.section_name] = new_sec
            i+=1
        all_params = []
        for sec in self.sections.values():
            for par_name, par in sec.params.items():
                all_params.append(par_name)
                setattr(self, par_name, par.value)

    def __str__(self) -> str:
        return "\n".join([str(s) for s in self.sections.values()])

    def __repr__(self) -> str:
        return self.name

    def save_changes(self):
        """docstring"""
        for sec in self.sections.values():
            for par_name in sec.params.keys():
                sec.params[par_name].value = getattr(self, par_name)


class dHybridRinput(File):
    """a class to read and write input files for dHybridR simulations"""
    header = f"""
        ! -------------------------------------------------------------------------------
        !   dHybridR input file
        !   Created by Keyan Gootkin's {__package__} v.{version(__package__.split('.')[0])} module
        ! -------------------------------------------------------------------------------
    """
    def __init__(self, path: str) -> None:
        if type(path)==list and type(path[0]==str): self.lines = path
        else:
            #init file properties
            File.__init__(self, path, master=dHybridRtemplate.path+"/input/input")
            #if doesn't exist, make a copy from master
            # if not self.exists: self.update()
            #read in the current file
            with open(self.path, 'r') as file: self.lines = file.read().split("\n")
        #get rid of empty lines
        self.lines = [l for l in self.lines if len(l.strip())>0]
        #read the remainder to set input file properties
        self.read()

    def __repr__(self): return "\n".join(self.lines)

    def read(self) -> None:
        assert is_input_header_boarder(self.lines[0]), "must start input file with header"
        self.sections = {}
        self.species = {}
        # go through input file header
        i=0
        while not is_input_header_boarder(self.lines[i+1]): i+=1
        i+=2
        # parse sections and species
        while i+1<len(self.lines):
            line = self.lines[i]
            #check if the start of a species description
            if is_input_species_section_header(line):
                #only accepting up to 9 species as of now
                sp_num = int(line.lower().split("species ")[-1][0])
                sp_start = i
                # species description ends when theres a section header that is not a
                # species section header for this species
                while not all([
                    is_input_section_header(self.lines[i+1]),
                    not is_input_species_section_header(self.lines[i+1], species=sp_num)
                ]):
                    if i+3>len(self.lines): break
                    i+=1
                sp_end = i+1
                #self.sp01 is the species we just read
                setattr(
                    self,
                    f"sp{str(sp_num).zfill(2)}",
                    (current_sp:=SpeciesInput(self.lines[sp_start:sp_end], sp_num))
                )
                self.species[sp_num] = current_sp
            #otherwise its the start of a normal section
            elif is_input_section_header(line) and not 'species' in line.lower():
                sec_start = i
                while not i+1==len(self.lines) and not is_input_section_header(self.lines[i+1]):
                    i+=1
                sec_end = i+1
                if sec_end-sec_start<3: continue
                new_sec = InputSection(self.lines[sec_start:sec_end])
                #if the section is all comments then ignore it
                if len(new_sec.lines)>3: self.sections[new_sec.section_name] = new_sec
            i+=1
        #set the section parameters to be self.parameter
        #(all repeat parameters should be in species objects)
        all_params = []
        for sec in self.sections.values():
            for par_name, par in sec.params.items():
                all_params.append(par_name)
                setattr(self, par_name, par.value)
        if len(unique(all_params))!=len(all_params):
            print("you have a duplicate parameter in your input file! >:-(\n\n", all_params)

    def save_changes(self) -> None:
        """docstring"""
        #set section parameters from self.parameters so that user can set the input parameters
        #to change the input section parameters and save those changes
        for sec in self.sections.values():
            for par_name in sec.params.keys():
                sec.params[par_name].value = getattr(self, par_name)
        for sp in self.species.values(): sp.save_changes()
        #seperate out sections to put before the species sections
        sections_not_all_species = [
            sec for sec in self.sections.values() if sec.section_name!='diag_species_total'
        ]
        #set lines
        head = self.header.split("\n")
        first_sections = "\n".join([str(sec) for sec in sections_not_all_species]).split("\n")
        species_sections  = "\n".join([str(sp) for sp in self.species.values()]).split("\n")
        final_section = str(self.sections['diag_species_total']).split("\n")
        self.lines = r_[
            head,
            first_sections,
            species_sections,
            final_section
        ]
        #write lines
        with open(self.path, 'w') as file: file.write("\n".join(self.lines))


class dHybridRout(File):
    """a class to read dHybridR out files"""
    def __init__(self, path: str) -> None:
        super().__init__(path)
        if self.exists:
            self.read()
            self.dt: datetime = self.end - self.start
            self.runtime: float = self.dt.seconds / 60. / 60.

    def read(self):
        File.read(self)
        self.start = datetime.now()
        self.end = datetime.now()
        for i,li in enumerate(self.lines):
            line: list[str] = li.strip().split()
            match line:
                case ['Run', 'started', start_day, '/', month, '/', year, 'at', hour, ':']:
                    current_day = int(start_day)
                    [minute, _, second, _, _] = self.lines[i+1].strip().split()
                    self.start = datetime(
                        int(year), int(month), int(start_day), int(hour), int(minute), int(second)
                    )
                    current = self.start
                case [
                    'Run', 'started', 
                    start_day, '/', month, '/', year, 'at',
                    hour, ':', minute, ':', second, ':', _
                ]:
                    current_day = int(start_day)
                    self.start = datetime(
                        int(year), int(month), int(start_day), int(hour), int(minute), int(second)
                    )
                    current = self.start
                case ['ITER', _, '-', hour, ':', minute, ':', second, ':', _]:
                    previous = current
                    if int(hour)<previous.hour: current_day += 1
                    current = datetime(
                        int(year), int(month), current_day, int(hour), int(minute), int(second)
                    )
                    self.end = current
                case ['ITER', _, '-', time]:
                    previous = current
                    hour, minute, second, _ = time.split(':')
                    if int(hour)<previous.hour: current_day += 1
                    current = datetime(
                        int(year), int(month), current_day, int(hour), int(minute), int(second)
                    )
                case ['Run', 'terminated', 'successfully', day, '/', month, '/', year, 'at']:
                    [hour, _, minute, _, second, _, _] = self.lines[i+1].strip().split()
                    self.end = datetime(
                        int(year), int(month), int(day), int(hour), int(minute), int(second)
                    )
                case [
                    'Run', 'terminated', 'successfully', 
                    day, '/', month, '/', year, 'at',
                    hour, ':', minute, ':', second, ':', _
                ]:
                    self.end = datetime(
                        int(year), int(month), int(day), int(hour), int(minute), int(second)
                    )
